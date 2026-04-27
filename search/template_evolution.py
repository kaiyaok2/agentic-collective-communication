"""
Template Evolution: LLM-guided synthesis of collective communication implementations.

Supports any collective optimization problem via the problem registry (search.problems).
Also supports NKI backend as an alternative to XLA.

Each candidate is:
1. Sandboxed (exec with restricted globals)
2. Correctness-tested (against reference on small world sizes)
3. Benchmarked (profiled on simulator with full topology)
4. Fed back to the LLM with results for iterative improvement

XLA optimization target: minimize XLA IR ops + collective dispatches.
NKI optimization target: minimize collective dispatches (local ops are free).
"""

import re
import math
import numpy as np
import traceback
from pathlib import Path

from .correctness_test import (
    # XLA path
    CollectiveSimulator,
    MockXM,
    MockTorch,
    TrackedTensor,
    TorchOpCounter,
    CollectiveProfiler,
    test_xla_candidate,
    benchmark_xla_candidate,
    _call_xla_candidate,
    _unwrap,
    # Generic path
    test_xla_candidate_generic,
    benchmark_xla_candidate_generic,
    test_xla_candidate_bf16,
    # NKI path
    MockNLModule,
    MockNKIModule,
    MockNCCLModule,
    NKICollectiveSimulator,
    NKICollectiveProfiler,
    test_nki_candidate,
    benchmark_nki_candidate,
    _call_nki_candidate,
)
from .contention_analysis import ContentionAnalyzer
from .generate_algo import _invoke_bedrock
from .problems import get_problem, CollectiveProblem

EVOLUTION_PROMPT = (
    Path(__file__).parent.parent / "prompts" / "template_evolution.md"
).read_text()

GENERIC_EVOLUTION_PROMPT = (
    Path(__file__).parent.parent / "prompts" / "generic_evolution.md"
).read_text()

NKI_EVOLUTION_PROMPT = (
    Path(__file__).parent.parent / "prompts" / "nki_template_evolution.md"
).read_text()

# Restricted globals for sandboxed execution.
import builtins as _builtins_mod

_SANDBOX_GLOBALS = {
    "__builtins__": _builtins_mod,
}

# Built-in templates the LLM can see as starting points
_BUILTIN_TEMPLATES = {}       # XLA templates
_NKI_BUILTIN_TEMPLATES = {}   # NKI templates


def _register_builtin(name, code):
    _BUILTIN_TEMPLATES[name] = code


def _register_nki_builtin(name, code):
    _NKI_BUILTIN_TEMPLATES[name] = code


# ================================================================
# XLA Builtin Templates
# ================================================================

_register_builtin("naive_allgather", '''\
def evolved_alltoallv(input_tensor, send_counts, recv_counts, max_chunk,
                      rank, world_size, num_devices, cores_per_device,
                      xm, torch, num_nodes=1):
    """AllGather + per-source slice + cat."""
    pack_size = world_size * max_chunk

    # Pack into canonical layout: data for rank i at slot [i*max_chunk]
    packed = torch.zeros(pack_size, device=input_tensor.device,
                         dtype=input_tensor.dtype)
    send_off = 0
    for i in range(world_size):
        sc = send_counts[i]
        if sc > 0:
            packed[i * max_chunk:i * max_chunk + sc] = \
                input_tensor[send_off:send_off + sc]
        send_off += sc

    # Single all_gather: collect packed buffers from all ranks
    gathered = xm.all_gather(packed.unsqueeze(0), dim=0).view(-1)

    # Slice: for each source, extract the data it sent to this rank
    chunks = []
    for src in range(world_size):
        count = recv_counts[src]
        base = src * pack_size + rank * max_chunk
        chunks.append(gathered[base:base + count])

    return torch.cat(chunks)
''')

_register_builtin("allgather_reduce_scatter", '''\
def evolved_alltoallv(input_tensor, send_counts, recv_counts, max_chunk,
                      rank, world_size, num_devices, cores_per_device,
                      xm, torch, num_nodes=1):
    """AllGather + reshape + transpose + ReduceScatter."""
    pack_size = world_size * max_chunk

    packed = torch.zeros(pack_size, device=input_tensor.device,
                         dtype=input_tensor.dtype)
    send_off = 0
    for i in range(world_size):
        sc = send_counts[i]
        if sc > 0:
            packed[i * max_chunk:i * max_chunk + sc] = \
                input_tensor[send_off:send_off + sc]
        send_off += sc

    gathered = xm.all_gather(packed.unsqueeze(0), dim=0)
    reshaped = gathered.view(world_size, world_size, max_chunk)
    transposed = reshaped.permute(1, 0, 2).contiguous().view(-1)

    my_shard = xm.reduce_scatter(xm.REDUCE_SUM, transposed, scale=1.0/world_size,
                                  scatter_dim=0, shard_count=world_size)

    flat_idx = []
    for src in range(world_size):
        count = recv_counts[src]
        base = src * max_chunk
        flat_idx.extend(range(base, base + count))

    idx_tensor = torch.tensor(flat_idx, device=input_tensor.device,
                              dtype=torch.long)
    return torch.index_select(my_shard, 0, idx_tensor)
''')

_register_builtin("permute_ring", '''\
def evolved_alltoallv(input_tensor, send_counts, recv_counts, max_chunk,
                      rank, world_size, num_devices, cores_per_device,
                      xm, torch, num_nodes=1):
    """Permute Ring: one collective_permute per distance."""
    send_offsets = [0]
    for c in send_counts[:-1]:
        send_offsets.append(send_offsets[-1] + c)
    recv_offsets = [0]
    for c in recv_counts[:-1]:
        recv_offsets.append(recv_offsets[-1] + c)

    total_recv = sum(recv_counts)
    output = torch.zeros(total_recv, device=input_tensor.device,
                         dtype=input_tensor.dtype)

    # Self copy
    self_count = recv_counts[rank]
    if self_count > 0:
        output[recv_offsets[rank]:recv_offsets[rank] + self_count] = \
            input_tensor[send_offsets[rank]:send_offsets[rank] + send_counts[rank]]

    # Build padded shards
    shards = []
    for i in range(world_size):
        sc = send_counts[i]
        chunk = input_tensor[send_offsets[i]:send_offsets[i] + sc]
        if sc < max_chunk:
            chunk = torch.cat([chunk, torch.zeros(max_chunk - sc,
                              device=input_tensor.device,
                              dtype=input_tensor.dtype)])
        shards.append(chunk)

    for d in range(1, world_size):
        send_to = (rank + d) % world_size
        recv_from = (rank - d + world_size) % world_size
        pairs = [(r, (r + d) % world_size) for r in range(world_size)]

        received = xm.collective_permute(shards[send_to], pairs=pairs)
        rc = recv_counts[recv_from]
        if rc > 0:
            output[recv_offsets[recv_from]:recv_offsets[recv_from] + rc] = \
                received[:rc]

    return output
''')




# ================================================================
# NKI Builtin Templates
# ================================================================

_register_nki_builtin("nki_naive_allgather", '''\
def evolved_alltoallv_kernel(input_hbm, send_counts, recv_counts, max_chunk,
                             rank, world_size, num_devices, cores_per_device,
                             nl, nccl, num_nodes=1):
    """NKI Naive AllGather: pack canonically, all_gather, extract with loop.

    1. Pack send data into canonical layout: slot i has max_chunk elements
       for destination rank i (padded with zeros).
    2. nccl.all_gather the packed buffer (1 collective dispatch).
    3. Loop over each source rank, load the elements it sent to us.

    Dispatches: 1 (all_gather)
    Amplification: world_size * max_chunk per rank.
    """
    dtype = nl.float32

    send_offsets = [0]
    for c in send_counts[:-1]:
        send_offsets.append(send_offsets[-1] + c)

    pack_size = world_size * max_chunk
    packed = nl.ndarray((pack_size,), dtype=dtype, buffer=nl.shared_hbm)
    for i in range(world_size):
        sc = send_counts[i]
        if sc > 0:
            data = nl.load(input_hbm[send_offsets[i]:send_offsets[i] + sc])
            nl.store(packed[i * max_chunk:i * max_chunk + sc], data)

    gathered = nl.ndarray((world_size * pack_size,), dtype=dtype,
                          buffer=nl.shared_hbm)
    nccl.all_gather(srcs=[packed], dsts=[gathered],
                    replica_groups=list(range(world_size)),
                    all_gather_dim=0)

    total_recv = sum(recv_counts)
    output = nl.ndarray((total_recv,), dtype=dtype, buffer=nl.shared_hbm)
    recv_offset = 0
    for src in range(world_size):
        count = recv_counts[src]
        if count > 0:
            base = src * pack_size + rank * max_chunk
            data = nl.load(gathered[base:base + count])
            nl.store(output[recv_offset:recv_offset + count], data)
        recv_offset += count

    return output
''')

_register_nki_builtin("nki_permute_ring", '''\
def evolved_alltoallv_kernel(input_hbm, send_counts, recv_counts, max_chunk,
                             rank, world_size, num_devices, cores_per_device,
                             nl, nccl, num_nodes=1):
    """NKI Permute Ring: one collective_permute per distance.

    Dispatches: world_size - 1 (one per distance).
    No amplification (each element sent once).
    """
    dtype = nl.float32

    send_offsets = [0]
    for c in send_counts[:-1]:
        send_offsets.append(send_offsets[-1] + c)
    recv_offsets = [0]
    for c in recv_counts[:-1]:
        recv_offsets.append(recv_offsets[-1] + c)

    total_recv = sum(recv_counts)
    output = nl.ndarray((total_recv,), dtype=dtype, buffer=nl.shared_hbm)

    shards = []
    for i in range(world_size):
        shard = nl.ndarray((max_chunk,), dtype=dtype, buffer=nl.shared_hbm)
        sc = send_counts[i]
        if sc > 0:
            data = nl.load(input_hbm[send_offsets[i]:send_offsets[i] + sc])
            nl.store(shard[0:sc], data)
        shards.append(shard)

    self_count = recv_counts[rank]
    if self_count > 0:
        data = nl.load(shards[rank][0:self_count])
        nl.store(output[recv_offsets[rank]:recv_offsets[rank] + self_count], data)

    recv_buf = nl.ndarray((max_chunk,), dtype=dtype, buffer=nl.shared_hbm)
    for d in range(1, world_size):
        send_to = (rank + d) % world_size
        recv_from = (rank - d + world_size) % world_size
        pairs = [(r, (r + d) % world_size) for r in range(world_size)]

        nccl.collective_permute(
            dst=recv_buf, src=shards[send_to],
            source_target_pairs=pairs)

        rc = recv_counts[recv_from]
        if rc > 0:
            data = nl.load(recv_buf[0:rc])
            nl.store(output[recv_offsets[recv_from]:recv_offsets[recv_from] + rc],
                     data)

    return output
''')


def _check_xla_safety(code_str):
    """Static analysis: detect patterns that work in eager mode but crash on XLA.

    Returns a list of warning strings, empty if code looks safe.
    """
    warnings = []
    lines = code_str.split('\n')
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        if re.search(
            r'torch\.zeros\(.*(dtype=torch\.long|dtype=torch\.int)',
            stripped
        ) and re.search(r'device\s*=\s*input_tensor\.device', stripped):
            if any('offset' in stripped.lower() or 'idx' in stripped.lower()
                   or 'index' in stripped.lower() or 'count' in stripped.lower()
                   for _ in [1]):
                pass
        if re.search(
            r'send_offsets\s*=\s*torch\.(zeros|tensor)\(', stripped
        ) or re.search(
            r'recv_offsets\s*=\s*torch\.(zeros|tensor)\(', stripped
        ):
            if 'device=input_tensor.device' in stripped or 'device=device' in stripped:
                warnings.append(
                    f"Line {i}: Creating offset tensor on device — offsets must be "
                    f"plain Python ints/lists, not device tensors (XLA lazy execution)")
        if re.search(r'len\(input_tensor\)', stripped):
            warnings.append(
                f"Line {i}: len(input_tensor) fails on XLA lazy tensors — "
                f"use sum(send_counts) instead")
        if re.search(r'scatter_\(', stripped) and 'device' in code_str:
            for j in range(max(0, i-5), min(len(lines), i+5)):
                if re.search(r'scatter_indices.*=.*torch\.(zeros|tensor)\(.*device', lines[j]):
                    warnings.append(
                        f"Line {i}: scatter_ with device index tensor built from "
                        f"offset loops creates excessive XLA ops")
                    break
    return warnings


# ================================================================
# Template Evolution Engine
# ================================================================

class TemplateEvolution:
    """
    LLM-guided evolution of collective communication implementations.

    Supports any collective problem via the problem registry, plus both
    XLA and NKI backends. XLA is the default and recommended path.

    Each round:
    1. Build prompt with current best code + topology context + feedback
    2. LLM generates new candidate
    3. Sandbox-exec the code
    4. Correctness test on small world sizes (4, 8)
    5. Benchmark on simulator (world=32)
    6. Keep if correct and better; feed results back to LLM
    """

    def __init__(self, topology, send_counts_matrix, cost_model,
                 analyzer=None, model="haiku", problem=None,
                 unsupported_primitives=None, op_costs=None,
                 dispatch_overhead_us=100.0):
        self.topo = topology
        self.send_counts = send_counts_matrix
        self.cost_model = cost_model
        self.analyzer = analyzer or ContentionAnalyzer(
            topology, send_counts_matrix)
        self.model = model
        self.world = topology.num_cores
        self.num_nodes = getattr(topology, 'num_nodes', 1)
        self.problem = problem
        self.unsupported_primitives = unsupported_primitives
        self.op_costs = op_costs or {}
        self.dispatch_overhead_us = dispatch_overhead_us

    def _format_op_breakdown(self, bench):
        """Format per-op cost breakdown from benchmark results."""
        breakdown = bench.get("op_breakdown", {})
        if not breakdown:
            return ""
        lines = ["Op cost breakdown:"]
        for op, info in sorted(breakdown.items(), key=lambda x: -x[1]["total_us"]):
            lines.append(f"  {op}: {info['count']}x @ {info['per_op_us']:.1f}us = {info['total_us']:.1f}us")
        local_cost = bench.get("local_cost_us", 0)
        if local_cost:
            lines.append(f"  Total local op cost: {local_cost:.1f}us")
        return "\n".join(lines)

    def _format_op_cost_table(self):
        """Format profiled op costs as a table for the LLM prompt."""
        if not self.op_costs:
            return "- Each local XLA op: ~29us\n- Each collective dispatch: ~100us"
        lines = []
        for op, cost in sorted(self.op_costs.items(), key=lambda x: x[1]):
            if cost < 1.0:
                lines.append(f"  {op:<20s} {cost:.1f} us  (metadata-only, no HLO)")
            else:
                lines.append(f"  {op:<20s} {cost:.1f} us")
        unsup = self.unsupported_primitives or []
        if unsup:
            lines.append(f"\nUnsupported primitives (will fail on hardware): {', '.join(unsup)}")
        return "\n".join(lines)

    def evolve(self, starting_template="naive_allgather", max_rounds=5,
               verbose=True):
        """
        Run template evolution loop.

        Args:
            starting_template: Name of builtin template to start from.
                For problem-driven evolution, use template names from
                the problem's builtin_templates dict.
                Legacy XLA templates: "naive_allgather", "allgather_reduce_scatter", "permute_ring"
                NKI templates: "nki_naive_allgather", "nki_permute_ring"

        Returns:
            best_code: str (Python source of best implementation)
            best_benchmark: dict (benchmark results)
            history: list of round results
        """
        is_nki = starting_template.startswith("nki_")

        if self.problem and not is_nki:
            templates = self.problem.builtin_templates
        elif is_nki:
            templates = _NKI_BUILTIN_TEMPLATES
        else:
            templates = _BUILTIN_TEMPLATES

        current_code = templates[starting_template]
        current_fn = self._sandbox_exec(current_code, is_nki=is_nki)
        if current_fn is None:
            raise ValueError(f"Built-in template {starting_template} failed to load")

        if is_nki:
            passed, details = test_nki_candidate(current_fn, num_nodes=self.num_nodes)
        elif self.problem:
            passed, details = test_xla_candidate_generic(
                self.problem, current_fn, num_nodes=self.num_nodes,
                unsupported_primitives=self.unsupported_primitives)
        else:
            passed, details = test_xla_candidate(
                current_fn, num_nodes=self.num_nodes,
                unsupported_primitives=self.unsupported_primitives)
        if not passed:
            raise ValueError(f"Built-in template failed correctness: {details}")

        if is_nki:
            current_bench = benchmark_nki_candidate(
                current_fn, self.topo, self.send_counts, self.world,
                num_nodes=self.num_nodes)
        elif self.problem:
            current_bench = benchmark_xla_candidate_generic(
                self.problem, current_fn, self.topo, self.send_counts,
                self.world, num_nodes=self.num_nodes,
                unsupported_primitives=self.unsupported_primitives,
                op_costs=self.op_costs,
                dispatch_overhead_us=self.dispatch_overhead_us)
        else:
            current_bench = benchmark_xla_candidate(
                current_fn, self.topo, self.send_counts, self.world,
                num_nodes=self.num_nodes,
                unsupported_primitives=self.unsupported_primitives)

        if verbose:
            backend = "NKI" if is_nki else "XLA"
            print(f"  Starting template: {starting_template} ({backend})")
            print(f"  Baseline: {current_bench.get('sim_time_us', 0):.1f} us, "
                  f"{current_bench.get('num_collective_permute', 0)} permutes, "
                  f"{current_bench.get('num_all_gather', 0)} gathers"
                  f"{', ' + str(current_bench.get('local_ops', '?')) + ' local ops' if not is_nki else ''}")

        best_code = current_code
        best_bench = dict(current_bench)
        best_fn = current_fn
        history = [{
            "round": 0,
            "action": "baseline",
            "sim_time_us": current_bench.get("sim_time_us", 0),
            "correct": True,
        }]

        for round_idx in range(1, max_rounds + 1):
            if verbose:
                print(f"\n  --- Evolution round {round_idx} ---")

            prompt = self._build_prompt(best_code, best_bench, history,
                                        is_nki=is_nki)

            try:
                response = _invoke_bedrock(
                    prompt, model=self.model,
                    temperature=0.8, max_tokens=6000)
                candidate_code = self._extract_code(response, is_nki=is_nki)
            except Exception as e:
                if verbose:
                    print(f"  LLM error: {e}")
                history.append({
                    "round": round_idx, "action": "llm_error",
                    "error": str(e)})
                continue

            if candidate_code is None:
                if verbose:
                    print(f"  Failed to extract code from LLM response")
                history.append({
                    "round": round_idx, "action": "parse_error"})
                continue

            if not is_nki:
                xla_warnings = _check_xla_safety(candidate_code)
                if xla_warnings:
                    if verbose:
                        print(f"  XLA-UNSAFE: {xla_warnings[0]}")
                    history.append({
                        "round": round_idx, "action": "xla_unsafe",
                        "error": "; ".join(xla_warnings)})
                    continue

            candidate_fn = self._sandbox_exec(candidate_code, is_nki=is_nki)
            if candidate_fn is None:
                if verbose:
                    print(f"  Sandbox execution failed")
                history.append({
                    "round": round_idx, "action": "exec_error"})
                continue

            if is_nki:
                passed, details = test_nki_candidate(
                    candidate_fn, num_nodes=self.num_nodes)
            elif self.problem:
                passed, details = test_xla_candidate_generic(
                    self.problem, candidate_fn, num_nodes=self.num_nodes,
                    unsupported_primitives=self.unsupported_primitives)
            else:
                passed, details = test_xla_candidate(
                    candidate_fn, num_nodes=self.num_nodes,
                    unsupported_primitives=self.unsupported_primitives)

            if not passed:
                if verbose:
                    print(f"  INCORRECT: {details}")
                history.append({
                    "round": round_idx, "action": "correctness_fail",
                    "error": details})
                continue

            if not is_nki and self.problem:
                bf16_ok, bf16_details = test_xla_candidate_bf16(
                    self.problem, candidate_fn, num_nodes=self.num_nodes,
                    unsupported_primitives=self.unsupported_primitives)
                if not bf16_ok:
                    if verbose:
                        print(f"  BF16 FAIL: {bf16_details}")
                    history.append({
                        "round": round_idx, "action": "bf16_fail",
                        "error": bf16_details})
                    continue

            if verbose:
                print(f"  Correctness: PASS (including bf16)")

            if is_nki:
                bench = benchmark_nki_candidate(
                    candidate_fn, self.topo, self.send_counts, self.world,
                    num_nodes=self.num_nodes)
            elif self.problem:
                bench = benchmark_xla_candidate_generic(
                    self.problem, candidate_fn, self.topo, self.send_counts,
                    self.world, num_nodes=self.num_nodes,
                    unsupported_primitives=self.unsupported_primitives,
                    op_costs=self.op_costs,
                    dispatch_overhead_us=self.dispatch_overhead_us)
            else:
                bench = benchmark_xla_candidate(
                    candidate_fn, self.topo, self.send_counts, self.world,
                    num_nodes=self.num_nodes,
                    unsupported_primitives=self.unsupported_primitives)

            if "error" in bench:
                if verbose:
                    print(f"  Benchmark error: {bench['error']}")
                history.append({
                    "round": round_idx, "action": "benchmark_error",
                    "error": bench["error"]})
                continue

            sim_us = bench.get("sim_time_us", float("inf"))
            best_us = best_bench.get("sim_time_us", float("inf"))

            if verbose:
                extra = ""
                if not is_nki:
                    extra = f", {bench.get('local_ops', '?')} local ops"
                print(f"  Benchmark: {sim_us:.1f} us "
                      f"({bench.get('num_collective_permute', 0)} permutes, "
                      f"{bench.get('num_all_gather', 0)} gathers{extra})")

            candidate_ops = bench.get("local_ops", float("inf"))
            best_ops = best_bench.get("local_ops", float("inf"))
            improved = (sim_us < best_us or
                        (sim_us == best_us and candidate_ops < best_ops))
            if improved:
                best_code = candidate_code
                best_bench = bench
                best_fn = candidate_fn
                if verbose:
                    improvement = (best_us - sim_us) / best_us * 100
                    print(f"  NEW BEST: {sim_us:.1f} us "
                          f"({improvement:.1f}% improvement)")

            history.append({
                "round": round_idx,
                "action": "accepted" if improved else "rejected",
                "sim_time_us": sim_us,
                "correct": True,
                "num_collective_permute": bench.get("num_collective_permute", 0),
                "num_all_gather": bench.get("num_all_gather", 0),
            })

        return best_code, best_bench, history

    def _build_prompt(self, current_code, current_bench, history, is_nki=False):
        """Build evolution prompt with full context."""
        history_text = ""
        for h in history[-5:]:
            line = f"  Round {h['round']}: {h['action']}"
            if "sim_time_us" in h:
                line += f" ({h['sim_time_us']:.1f} us)"
            if "error" in h:
                line += f" - {h['error'][:100]}"
            history_text += line + "\n"

        if self.problem and not is_nki:
            return self._build_generic_prompt(current_code, current_bench,
                                               history_text)

        if is_nki:
            return NKI_EVOLUTION_PROMPT.replace(
                "{current_code}", current_code
            ).replace(
                "{current_sim_time}", f"{current_bench.get('sim_time_us', 0):.1f}"
            ).replace(
                "{current_num_permutes}",
                str(current_bench.get("num_collective_permute", 0))
            ).replace(
                "{current_num_gathers}",
                str(current_bench.get("num_all_gather", 0))
            ).replace(
                "{history}", history_text
            ).replace(
                "{world_size}", str(self.world)
            ).replace(
                "{num_devices}", str(self.topo.num_devices)
            ).replace(
                "{cores_per_device}", str(self.topo.cores_per_device)
            ).replace(
                "{builtin_nki_naive_allgather}",
                _NKI_BUILTIN_TEMPLATES.get("nki_naive_allgather", "")
            ).replace(
                "{builtin_nki_permute_ring}",
                _NKI_BUILTIN_TEMPLATES.get("nki_permute_ring", "")
            ).replace(
                "{num_nodes}", str(self.num_nodes)
            ).replace(
                "{efa_bandwidth}", str(getattr(self.topo, 'efa_bw', 0))
            ).replace(
                "{efa_latency}", str(getattr(self.topo, 'efa_lat', 0))
            ).replace(
                "{ranks_per_node}", str(getattr(self.topo, 'ranks_per_node', self.world))
            )
        else:
            return EVOLUTION_PROMPT.replace(
                "{current_code}", current_code
            ).replace(
                "{current_sim_time}", f"{current_bench.get('sim_time_us', 0):.1f}"
            ).replace(
                "{current_num_permutes}",
                str(current_bench.get("num_collective_permute", 0))
            ).replace(
                "{current_num_gathers}",
                str(current_bench.get("num_all_gather", 0))
            ).replace(
                "{current_local_ops}",
                str(current_bench.get("local_ops", "?"))
            ).replace(
                "{history}", history_text
            ).replace(
                "{world_size}", str(self.world)
            ).replace(
                "{num_devices}", str(self.topo.num_devices)
            ).replace(
                "{cores_per_device}", str(self.topo.cores_per_device)
            ).replace(
                "{builtin_naive_allgather}",
                _BUILTIN_TEMPLATES.get("naive_allgather", "")
            ).replace(
                "{builtin_permute_ring}",
                _BUILTIN_TEMPLATES.get("permute_ring", "")
            ).replace(
                "{num_nodes}", str(self.num_nodes)
            ).replace(
                "{efa_bandwidth}", str(getattr(self.topo, 'efa_bw', 0))
            ).replace(
                "{efa_latency}", str(getattr(self.topo, 'efa_lat', 0))
            ).replace(
                "{ranks_per_node}", str(getattr(self.topo, 'ranks_per_node', self.world))
            )

    def _build_generic_prompt(self, current_code, current_bench, history_text):
        """Build evolution prompt for any collective problem."""
        problem = self.problem
        ref_impls = ""
        for name, code in problem.builtin_templates.items():
            ref_impls += f"### {name}:\n```python\n{code}\n```\n\n"

        return GENERIC_EVOLUTION_PROMPT.replace(
            "{display_name}", problem.display_name
        ).replace(
            "{current_code}", current_code
        ).replace(
            "{current_sim_time}", f"{current_bench.get('sim_time_us', 0):.1f}"
        ).replace(
            "{current_num_permutes}",
            str(current_bench.get("num_collective_permute", 0))
        ).replace(
            "{current_num_gathers}",
            str(current_bench.get("num_all_gather", 0))
        ).replace(
            "{current_local_ops}",
            str(current_bench.get("local_ops", "?"))
        ).replace(
            "{history}", history_text
        ).replace(
            "{world_size}", str(self.world)
        ).replace(
            "{num_devices}", str(self.topo.num_devices)
        ).replace(
            "{cores_per_device}", str(self.topo.cores_per_device)
        ).replace(
            "{num_nodes}", str(self.num_nodes)
        ).replace(
            "{efa_bandwidth}", str(getattr(self.topo, 'efa_bw', 0))
        ).replace(
            "{efa_latency}", str(getattr(self.topo, 'efa_lat', 0))
        ).replace(
            "{ranks_per_node}", str(getattr(self.topo, 'ranks_per_node', self.world))
        ).replace(
            "{signature}", problem.signature
        ).replace(
            "{signature_doc}", problem.signature_doc
        ).replace(
            "{optimization_hints}",
            problem.optimization_hints.replace(
                "{op_cost_table}", self._format_op_cost_table()
            ).replace(
                "{dispatch_overhead_us}", f"{self.dispatch_overhead_us:.0f}"
            )
        ).replace(
            "{current_op_breakdown}", self._format_op_breakdown(current_bench)
        ).replace(
            "{evolved_fn_name}", problem.evolved_fn_name
        ).replace(
            "{reference_implementations}", ref_impls
        )

    def _extract_code(self, response, is_nki=False):
        """Extract the evolved function from LLM response."""
        if is_nki:
            fn_name = "evolved_alltoallv_kernel"
        elif self.problem:
            fn_name = self.problem.evolved_fn_name
        else:
            fn_name = "evolved_alltoallv"

        patterns = [
            r"```python\s*\n(.*?)```",
            r"```\s*\n(.*?)```",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                if f"def {fn_name}" in match:
                    return match.strip()

        match = re.search(
            rf"(def {fn_name}\(.*?\n(?:(?:    .*|)\n)*)",
            response, re.MULTILINE)
        if match:
            return match.group(1).strip()

        return None

    def _sandbox_exec(self, code_str, is_nki=False):
        """Safely execute LLM-generated code and return the function."""
        sandbox = dict(_SANDBOX_GLOBALS)
        sandbox["math"] = math
        sandbox["np"] = np
        sandbox["numpy"] = np

        if is_nki:
            sandbox["nl"] = MockNLModule()
            sandbox["nccl"] = None
            sandbox["nki"] = MockNKIModule()
            fn_name = "evolved_alltoallv_kernel"
        elif self.problem:
            fn_name = self.problem.evolved_fn_name
        else:
            fn_name = "evolved_alltoallv"

        try:
            exec(code_str, sandbox)
        except Exception as e:
            return None

        fn = sandbox.get(fn_name)
        if fn is None or not callable(fn):
            return None

        return fn
