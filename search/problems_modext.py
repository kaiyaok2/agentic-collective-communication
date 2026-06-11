"""Problem registrations for model-extension collectives — v2.

Templates simplified to avoid matmul (MockTorch sandbox does not support
TrackedTensor @ TrackedTensor). The microbatch dispatch pattern --- per-mb
mark_step vs bundled --- is what the agent searches; the actual matmul is
a real-training detail that the simulator scores via the broader per-op
cost table, not via in-template emulation.

These three new problems all share the structural shape: M microbatches,
K collectives per microbatch. The per_mb template issues M*K dispatches
across M mark_step graphs; the bundled template issues K dispatches in
one mark_step graph.
"""

import torch

from .problems import CollectiveProblem, register_problem


# ================================================================
# Problem 6: Pipeline-parallel cross-stage send/recv
# ================================================================

def _pp_send_recv_reference(activations, src_stage, half, world_size):
    """Reference: dst_stage receives the paired src_stage rank's activation.
    src_stage receives its own activations back (R9-ter: src_stage returns
    own AR result — matches what xm.all_reduce(REDUCE_SUM, masked_buf)
    naturally produces when only the src_stage rank populates its slot).
    In real training, src_stage's output is unused; this convention just
    makes the correctness gate consistent with the baseline."""
    M = len(activations[0])
    outputs = [[None] * M for _ in range(world_size)]
    for m in range(M):
        for r in range(world_size):
            stage = 0 if r < half else 1
            if stage == src_stage:
                outputs[r][m] = activations[r][m].clone()
            else:
                pair = r - half if stage == 1 else r + half
                outputs[r][m] = activations[pair][m].clone()
    return outputs


def _pp_send_recv_generate_test_case(world_size, pattern="uniform",
                                      M=4, B=1, S=8, DM=16, seed=0):
    torch.manual_seed(seed)
    half = world_size // 2
    per_rank_args = []
    for r in range(world_size):
        in_src = r < half
        acts = [torch.randn(B, S, DM) * 0.01 if in_src else torch.zeros(B, S, DM)
                for _ in range(M)]
        per_rank_args.append({"activations": acts, "src_stage": 0,
                              "half": half, "M": M})
    expected = []
    for r in range(world_size):
        if r < half:
            # src_stage: AR-with-mask returns this src rank's own activations
            # (R9-ter: align inline expected with per_mb baseline natural output).
            expected.append([per_rank_args[r]["activations"][m].clone()
                             for m in range(M)])
        else:
            expected.append([per_rank_args[r - half]["activations"][m].clone()
                             for m in range(M)])
    return {"per_rank_args": per_rank_args, "shared_args": {}, "expected": expected}


def _pp_send_recv_call_candidate(candidate_fn, rank_args, shared_args,
                                  rank, world_size, num_devices, cpd,
                                  xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(
        rank_args["activations"], rank_args["src_stage"],
        rank_args["half"], rank_args["M"],
        rank, world_size, num_devices, cpd, xm_mock, torch_mock,
        num_nodes=num_nodes)


_PP_SIGNATURE = """\
def evolved_pp_send_recv(activations, src_stage, half, M, rank, world_size,
                         num_devices, cores_per_device, xm, torch, num_nodes=1):"""

_PP_SIGNATURE_DOC = """\
    Args:
        activations: list[Tensor] of length M, each shape (B, S, DM).
            Only src_stage ranks have meaningful data.
        src_stage: int (0 or 1).
        half: int. world_size // 2.
        M: int microbatches.
        rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes.

    Returns:
        list[Tensor] of length M, where outputs[m] has the same shape as
        activations[m]. The return type MUST be a flat Python list of M
        tensors (NOT a single stacked tensor; NOT a nested list).
        Common pitfall: if any internal step produces a bundled or
        stacked tensor, the final return value must still be restructured
        back into the per-microbatch list[Tensor] container before
        returning."""

_PP_BUILTINS = {}

_PP_BUILTINS["per_mb"] = '''\
def evolved_pp_send_recv(activations, src_stage, half, M, rank, world_size,
                         num_devices, cores_per_device, xm, torch, num_nodes=1):
    """per_mb_loop_AR: one masked all_reduce per microbatch (M dispatches, M graphs)."""
    stage = 0 if rank < half else 1
    pair_id = rank if stage == 0 else rank - half
    a0 = activations[0]
    outs = []
    for m in range(M):
        buf = torch.zeros(half, *a0.shape, dtype=a0.dtype)
        if stage == src_stage:
            buf[pair_id] = activations[m]
        ared = xm.all_reduce(xm.REDUCE_SUM, buf)
        outs.append(ared[pair_id])
        # mark_step boundary intrinsic in real training; omitted in sandbox
    return outs
'''


_PP_HINTS = """\
### Constraints:
- The baseline 'per_mb' issues xm.mark_step() between microbatches; XLA
  cannot fuse work across mark_step boundaries.
- HBM is a finite resource: any (half, M, *act_shape) buffer scales
  linearly with M, and on 7-node Trainium each Neuron core has ~32GB HBM
  shared between two cores."""

register_problem(CollectiveProblem(
    name="pp_send_recv",
    display_name="Pipeline-parallel cross-stage send/recv",
    evolved_fn_name="evolved_pp_send_recv",
    signature=_PP_SIGNATURE,
    signature_doc=_PP_SIGNATURE_DOC,
    reference_fn=_pp_send_recv_reference,
    generate_test_case=_pp_send_recv_generate_test_case,
    call_candidate=_pp_send_recv_call_candidate,
    builtin_templates=_PP_BUILTINS,
    optimization_hints=_PP_HINTS,
))


# ================================================================
# Problem 7: TP MLP — abstracted to "many AR per microbatch"
# ================================================================
#
# Real TP MLP fires one AR per layer per microbatch after the row-parallel
# matmul. We can't run a matmul in MockTorch, but we can mirror the
# dispatch pattern by treating each layer's "partial output" as already
# computed and AR-ing it. The agent's job is to bundle ARs across M
# microbatches, which is the same structural choice as in real training.

def _tp_mlp_reference(partials_per_rank, world_size):
    """Reference: AR-sum each (microbatch, layer) partial across ranks."""
    M = len(partials_per_rank[0])
    L_count = len(partials_per_rank[0][0])
    outputs = [[None] * M for _ in range(world_size)]
    for m in range(M):
        out_per_layer = []
        for L in range(L_count):
            summed = partials_per_rank[0][m][L].clone()
            for r in range(1, world_size):
                summed = summed + partials_per_rank[r][m][L]
            out_per_layer.append(summed)
        for r in range(world_size):
            # Each rank produces the same M-sequence of per-layer ARs.
            outputs[r][m] = out_per_layer
    return outputs


def _tp_mlp_generate_test_case(world_size, pattern="uniform",
                                M=4, N_LAYERS=2, B=1, S=8, DM=16, seed=0):
    torch.manual_seed(seed)
    per_rank_args = []
    for r in range(world_size):
        partials = []
        for m in range(M):
            partials.append([torch.randn(B, S, DM) * 0.01 for _ in range(N_LAYERS)])
        per_rank_args.append({"partials": partials, "M": M, "N_LAYERS": N_LAYERS})
    # reference output: each rank gets full AR for each (m, L).
    full_partials = [[torch.zeros(B, S, DM) for _ in range(N_LAYERS)] for _ in range(M)]
    for m in range(M):
        for L in range(N_LAYERS):
            for r in range(world_size):
                full_partials[m][L] = full_partials[m][L] + per_rank_args[r]["partials"][m][L]
    expected = [[full_partials[m] for m in range(M)] for _ in range(world_size)]
    return {"per_rank_args": per_rank_args, "shared_args": {}, "expected": expected}


def _tp_mlp_call_candidate(candidate_fn, rank_args, shared_args,
                            rank, world_size, num_devices, cpd,
                            xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(
        rank_args["partials"], rank_args["M"], rank_args["N_LAYERS"],
        rank, world_size, num_devices, cpd, xm_mock, torch_mock,
        num_nodes=num_nodes)


_TP_SIGNATURE = """\
def evolved_tp_mlp(partials, M, N_LAYERS, rank, world_size,
                   num_devices, cores_per_device, xm, torch, num_nodes=1):"""

_TP_SIGNATURE_DOC = """\
    Args:
        partials: list[list[Tensor]] of shape [M][N_LAYERS]; partial sums per
            (microbatch, layer) computed locally (matmul output before TP AR).
        M, N_LAYERS: int.
        rank, world_size, ...: standard.

    Returns:
        list[list[Tensor]] with EXACT outer length M and EXACT inner length
        N_LAYERS, where outputs[m][L] has the same shape as partials[m][L].
        The return type MUST be a nested Python list of lists (NOT a flat
        list of length M*N_LAYERS; NOT a single stacked tensor).
        Common pitfall: if any internal step produces a bundled or stacked
        tensor, the final return value must still be restructured back into
        the nested [M][N_LAYERS] container before returning."""

_TP_BUILTINS = {}
_TP_BUILTINS["per_mb"] = '''\
def evolved_tp_mlp(partials, M, N_LAYERS, rank, world_size,
                   num_devices, cores_per_device, xm, torch, num_nodes=1):
    """per_mb_loop_AR: AR each layer's partial per-microbatch, mark_step between μbatches."""
    outs = []
    for m in range(M):
        per_layer = []
        for L in range(N_LAYERS):
            per_layer.append(xm.all_reduce(xm.REDUCE_SUM, partials[m][L]))
        outs.append(per_layer)
        # mark_step boundary intrinsic in real training; omitted in sandbox
    return outs
'''

_TP_HINTS = """\
### Constraints:
- The 'per_mb' baseline issues M*N_LAYERS AR dispatches across M mark_step
  graphs; XLA cannot fuse across mark_step boundaries.
- HBM is a finite resource: stacked buffers of shape
  (M*N_LAYERS, *partial_shape) scale linearly with M and N_LAYERS."""

register_problem(CollectiveProblem(
    name="tp_mlp",
    display_name="TP MLP with microbatching",
    evolved_fn_name="evolved_tp_mlp",
    signature=_TP_SIGNATURE,
    signature_doc=_TP_SIGNATURE_DOC,
    reference_fn=_tp_mlp_reference,
    generate_test_case=_tp_mlp_generate_test_case,
    call_candidate=_tp_mlp_call_candidate,
    builtin_templates=_TP_BUILTINS,
    optimization_hints=_TP_HINTS,
))


# ================================================================
# Problem 8: FSDP sharded weight prefetch — abstracted to "many AG per microbatch"
# ================================================================
#
# Real FSDP fires one AG per layer per microbatch (per shard). We abstract
# to AG-ing a sharded tensor per (microbatch, layer); the agent's job is to
# either (a) recognise the weight is the same across microbatches and AG
# once, or (b) bundle the AGs across microbatches into one.

def _fsdp_prefetch_reference(shards_per_rank, world_size):
    """Reference: AG-concatenate the shard along dim 1 for each (m, L)."""
    M = len(shards_per_rank[0])
    L_count = len(shards_per_rank[0][0])
    outputs = [[None] * M for _ in range(world_size)]
    for m in range(M):
        out_per_layer = []
        for L in range(L_count):
            gathered = torch.cat([shards_per_rank[r][m][L]
                                  for r in range(world_size)], dim=-1)
            out_per_layer.append(gathered)
        for r in range(world_size):
            outputs[r][m] = out_per_layer
    return outputs


def _fsdp_prefetch_generate_test_case(world_size, pattern="uniform",
                                       M=4, N_LAYERS=2, shard_size=8, DM=16, seed=0):
    torch.manual_seed(seed)
    per_rank_args = []
    for r in range(world_size):
        # Same shard across microbatches (weight is fixed).
        shards = []
        for m in range(M):
            shards.append([torch.randn(DM, shard_size) * 0.01 for _ in range(N_LAYERS)])
        per_rank_args.append({"shards": shards, "M": M, "N_LAYERS": N_LAYERS})
    expected = _fsdp_prefetch_reference([a["shards"] for a in per_rank_args], world_size)
    return {"per_rank_args": per_rank_args, "shared_args": {}, "expected": expected}


def _fsdp_prefetch_call_candidate(candidate_fn, rank_args, shared_args,
                                   rank, world_size, num_devices, cpd,
                                   xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(
        rank_args["shards"], rank_args["M"], rank_args["N_LAYERS"],
        rank, world_size, num_devices, cpd, xm_mock, torch_mock,
        num_nodes=num_nodes)


_FSDP_SIGNATURE = """\
def evolved_fsdp_prefetch(shards, M, N_LAYERS, rank, world_size,
                          num_devices, cores_per_device, xm, torch, num_nodes=1):"""

_FSDP_SIGNATURE_DOC = """\
    Args:
        shards: list[list[Tensor]] of shape [M][N_LAYERS]; each shard is
            this rank's portion of the per-layer weight.
        M, N_LAYERS: int.
        rank, world_size, ...: standard.

    Returns:
        list[list[Tensor]] with EXACT outer length M and EXACT inner length
        N_LAYERS, where outputs[m][L] is the full all-gathered weight for
        layer L at microbatch m. The return type MUST be a nested Python
        list of lists (NOT a flat list of length M*N_LAYERS; NOT a single
        stacked tensor).
        Common pitfall: if any internal step produces a bundled or stacked
        tensor, the final return value must still be restructured back into
        the nested [M][N_LAYERS] container before returning."""

_FSDP_BUILTINS = {}
_FSDP_BUILTINS["per_mb"] = '''\
def evolved_fsdp_prefetch(shards, M, N_LAYERS, rank, world_size,
                          num_devices, cores_per_device, xm, torch, num_nodes=1):
    """per_mb_loop_AG: AG each layer's shard per-microbatch, mark_step between μbatches."""
    outs = []
    for m in range(M):
        per_layer = []
        for L in range(N_LAYERS):
            per_layer.append(xm.all_gather(shards[m][L], dim=-1))
        outs.append(per_layer)
        # mark_step boundary intrinsic in real training; omitted in sandbox
    return outs
'''

_FSDP_HINTS = """\
### Constraints:
- The 'per_mb' baseline issues M*N_LAYERS AG dispatches across M mark_step
  graphs; XLA cannot fuse across mark_step boundaries.
- HBM is finite: stacked buffers of shape (M*N_LAYERS, *shard_shape) or
  (N_LAYERS, *full_weight_shape) scale linearly with the bundling dimension."""

register_problem(CollectiveProblem(
    name="fsdp_prefetch",
    display_name="FSDP-style sharded weight prefetch with microbatching",
    evolved_fn_name="evolved_fsdp_prefetch",
    signature=_FSDP_SIGNATURE,
    signature_doc=_FSDP_SIGNATURE_DOC,
    reference_fn=_fsdp_prefetch_reference,
    generate_test_case=_fsdp_prefetch_generate_test_case,
    call_candidate=_fsdp_prefetch_call_candidate,
    builtin_templates=_FSDP_BUILTINS,
    optimization_hints=_FSDP_HINTS,
))


# ================================================================
# Problem 9: Llama transformer-block AR fusion (parallel-attention)
# ================================================================

def _llama_block_reference(attn_partials, mlp_partials, world_size):
    """Reference: AR-sum each (attn, mlp) partial across ranks.
    Returns the summed attn and mlp outputs per rank."""
    attn_sum = attn_partials[0].clone()
    mlp_sum  = mlp_partials[0].clone()
    for r in range(1, world_size):
        attn_sum = attn_sum + attn_partials[r]
        mlp_sum  = mlp_sum  + mlp_partials[r]
    # Each rank gets the same full summed output.
    return [[attn_sum, mlp_sum] for _ in range(world_size)]


def _llama_block_generate_test_case(world_size, pattern="uniform",
                                     B=1, S=8, DM=16, seed=0):
    torch.manual_seed(seed)
    attn_partials = [torch.randn(B, S, DM) * 0.01 for _ in range(world_size)]
    mlp_partials  = [torch.randn(B, S, DM) * 0.01 for _ in range(world_size)]
    per_rank_args = []
    for r in range(world_size):
        per_rank_args.append({
            "attn_partial": attn_partials[r],
            "mlp_partial":  mlp_partials[r],
        })
    expected = _llama_block_reference(attn_partials, mlp_partials, world_size)
    return {"per_rank_args": per_rank_args, "shared_args": {}, "expected": expected}


def _llama_block_call_candidate(candidate_fn, rank_args, shared_args,
                                 rank, world_size, num_devices, cpd,
                                 xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(
        rank_args["attn_partial"], rank_args["mlp_partial"],
        rank, world_size, num_devices, cpd,
        xm_mock, torch_mock, num_nodes=num_nodes)


_LB_SIGNATURE = """\
def evolved_llama_block(attn_partial, mlp_partial, rank, world_size,
                        num_devices, cores_per_device, xm, torch, num_nodes=1):"""

_LB_SIGNATURE_DOC = """\
    Args:
        attn_partial: (B, S, DM) row-parallel attention out_proj partial.
        mlp_partial:  (B, S, DM) row-parallel MLP down_proj partial.
        rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes:
            standard arguments.

    Returns:
        list [attn_full, mlp_full] where each is the AR-summed output.
"""

_LB_BUILTINS = {}

_LB_BUILTINS["sequential_2ar"] = '''\
def evolved_llama_block(attn_partial, mlp_partial, rank, world_size,
                        num_devices, cores_per_device, xm, torch, num_nodes=1):
    """Developer baseline: two sequential AR calls, one per primitive."""
    attn_full = xm.all_reduce(xm.REDUCE_SUM, attn_partial)
    mlp_full  = xm.all_reduce(xm.REDUCE_SUM, mlp_partial)
    return [attn_full, mlp_full]
'''


_LB_HINTS = """\
### Constraints:
- The sequential_2ar baseline issues 2 sequential ARs (one after attention out_proj,
  one after MLP down_proj). On this hardware Neuron does not auto-fuse
  collectives across data dependencies.
- The architectural prerequisite for ANY attention-MLP fusion (parallel
  attention-MLP, PaLM/GPT-J style) is that MLP consumes the same
  normalised input as attention rather than attention's output. This is
  the problem-level architectural choice the user has adopted; the
  collective-level question is what to do given that choice."""

register_problem(CollectiveProblem(
    name="llama_block_ar",
    display_name="Llama transformer-block AR fusion (parallel-attention)",
    evolved_fn_name="evolved_llama_block",
    signature=_LB_SIGNATURE,
    signature_doc=_LB_SIGNATURE_DOC,
    reference_fn=_llama_block_reference,
    generate_test_case=_llama_block_generate_test_case,
    call_candidate=_llama_block_call_candidate,
    builtin_templates=_LB_BUILTINS,
    optimization_hints=_LB_HINTS,
))
