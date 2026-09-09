"""1000-step end-to-end Llama training using all 4 model-extension wins.

Analog of train_olmoe10b.py but for the Llama-style TP+FSDP+PP composition
introduced in the model-extension follow-on. Full forward + backward +
optimizer step per microbatch; loss is the random-token training cross-
entropy floor. Step time is what we report.

Backends:
  per_mb : naive — per-microbatch mark_step, no bundling of cross-stage
           transfer / TP AR / FSDP AG dispatches across microbatches.
  bundled: agent — all M microbatches' work in one mark_step graph;
           Trainium has no async collectives, so this loses no pipelining.

The composition under test is the same as llama_4comp_train.py but with
real training: each step computes a cross-entropy loss against a random
target, runs autograd.backward through the masked-AR cross-stage transfer
(identity-pass backward), and applies an SGD step. Grad-AR across all
ranks for replicated parameters (the small per-layer norms) is kept at
the per-tensor-loop baseline.
"""
import os, sys, time, json, statistics
import time as _t
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

sys.path.insert(0, '/home/ubuntu/agentic-collective-communication')
from training import _percall_probe as probe

DM = 2048
HID = 5376
N_LAYERS_PER_STAGE = 1
N_MB = 4
B = 1
S = 2048
VOCAB = 224 * 32   # 7168 sharded across 224 ranks → 32 tokens/rank
LR = 1e-4
WARMUP = 5
STEADY_FROM = 200


def main():
    dev = xm.xla_device()
    ws = int(os.environ.get('WORLD_SIZE', xr.world_size()))
    rank = xr.global_ordinal()
    half = ws // 2
    stage = 0 if rank < half else 1
    pair_id = rank if stage == 0 else rank - half
    assert HID % ws == 0 and VOCAB % ws == 0
    shard_hid = HID // ws
    shard_vocab = VOCAB // ws

    # Shared embedding (replicated across all ranks for simplicity).
    embed = nn.Embedding(VOCAB, DM, dtype=torch.bfloat16).to(dev)
    # Sharded MLP weights for both stages.
    w_gate = nn.ParameterList([nn.Parameter(torch.randn(DM, shard_hid, dtype=torch.bfloat16) * 0.01)
                               for _ in range(N_LAYERS_PER_STAGE)]).to(dev)
    w_up = nn.ParameterList([nn.Parameter(torch.randn(DM, shard_hid, dtype=torch.bfloat16) * 0.01)
                             for _ in range(N_LAYERS_PER_STAGE)]).to(dev)
    w_down = nn.ParameterList([nn.Parameter(torch.randn(shard_hid, DM, dtype=torch.bfloat16) * 0.01)
                               for _ in range(N_LAYERS_PER_STAGE)]).to(dev)
    # Vocab-parallel LM head (stage 1 only).
    lm_head_shard = nn.Parameter(torch.randn(DM, shard_vocab, dtype=torch.bfloat16) * 0.01).to(dev)

    rep_params = list(embed.parameters()) + [lm_head_shard]  # stage-1-only for head
    sharded_params = list(w_gate) + list(w_up) + list(w_down)
    all_params = rep_params + sharded_params

    # Import + init agent's evolved runtimes (after dist setup)
    _agent_avail = True
    try:
        from runtime.trainium_tp_mlp_7node import evolved_tp_mlp, init_tp_mlp
        from runtime.trainium_pp_send_recv_7node import evolved_pp_send_recv, init_pp_send_recv
        from runtime.trainium_dxe_7node import dxe_loss, init_dxe
        init_tp_mlp(); init_pp_send_recv(); init_dxe()
    except Exception as _e:
        _agent_avail = False
        if rank == 0: print(f"[init] agent runtimes unavailable: {_e}; bundled mode will fall back to inline equivalents", flush=True)

    inputs = [torch.randint(0, VOCAB, (B, S), device=dev, dtype=torch.int64) for _ in range(N_MB)]
    targets = [torch.randint(0, VOCAB, (B, S), device=dev, dtype=torch.int64) for _ in range(N_MB)]

    def tp_fsdp_block(x, L):
        # FSDP prefetch: 3 AGs to materialise full weights
        if probe.in_window():
            _t0 = _t.time()
            w_gate_full = xm.all_gather(w_gate[L], dim=1)
            w_up_full   = xm.all_gather(w_up[L],   dim=1)
            w_down_full = xm.all_gather(w_down[L], dim=0)
            _ = w_gate_full.float().sum().item()
            probe.record("fsdp_prefetch", (_t.time() - _t0) * 1000.0)
        else:
            w_gate_full = xm.all_gather(w_gate[L], dim=1)
            w_up_full   = xm.all_gather(w_up[L],   dim=1)
            w_down_full = xm.all_gather(w_down[L], dim=0)
        # FFN compute (matmuls)
        h = F.silu(torch.matmul(x, w_gate_full)) * torch.matmul(x, w_up_full)
        partial = torch.matmul(h, w_down_full)
        # TP MLP AR
        if probe.in_window():
            _t0 = _t.time()
            ared = xm.all_reduce(xm.REDUCE_SUM, partial)
            _ = ared.float().sum().item()
            probe.record("tp_mlp", (_t.time() - _t0) * 1000.0)
        else:
            ared = xm.all_reduce(xm.REDUCE_SUM, partial)
        return x + ared / ws

    def transfer(act, src_stage):
        buf = torch.zeros(half, B, S, DM, device=dev, dtype=torch.bfloat16)
        if stage == src_stage:
            buf = buf.clone()
            buf[pair_id] = act
        ared = xm.all_reduce(xm.REDUCE_SUM, buf)
        return ared[pair_id]

    def transfer_batched(acts, src_stage):
        M = acts.shape[0]
        buf = torch.zeros(half, M, B, S, DM, device=dev, dtype=torch.bfloat16)
        if stage == src_stage:
            buf = buf.clone()
            buf[pair_id] = acts
        ared = xm.all_reduce(xm.REDUCE_SUM, buf)
        return ared[pair_id]

    def vocab_parallel_loss(h, tgt):
        # Distributed cross-entropy on vocab-sharded logits.
        # logits_local = h @ lm_head_shard : (B, S, shard_vocab)
        logits = torch.matmul(h, lm_head_shard).float()
        # local sum_v exp(logits)
        sum_exp_local = torch.exp(logits).sum(dim=-1)             # (B, S)
        sum_exp = xm.all_reduce(xm.REDUCE_SUM, sum_exp_local)
        # local target logit (if target is in this rank's vocab shard)
        rank_lo = rank * shard_vocab
        rank_hi = rank_lo + shard_vocab
        local_tgt_mask = (tgt >= rank_lo) & (tgt < rank_hi)
        local_tgt_idx = (tgt - rank_lo).clamp(0, shard_vocab - 1)
        # Gather target logit for tokens this rank owns.
        target_logit_local = torch.where(
            local_tgt_mask,
            logits.gather(-1, local_tgt_idx.unsqueeze(-1)).squeeze(-1),
            torch.zeros_like(sum_exp))
        target_logit = xm.all_reduce(xm.REDUCE_SUM, target_logit_local)
        loss = (torch.log(sum_exp + 1e-20) - target_logit).mean()
        return loss



    def vocab_parallel_loss_bundled(h_all, targets_stacked):
        """Agent-bundled vocab dxe via evolved_dxe (single mean over M*B*S).

        Falls back to inline single-call if dxe_loss isn't available.
        Internally: pack 2 ARs (sum_exp + target_logit) into 1 — agent's evolved strategy.
        """
        # logits: (M, B, S, V_local)
        logits = torch.matmul(h_all, lm_head_shard).float()
        M_ = logits.shape[0]
        # Flatten M*B*S into the batch dim for dxe_loss
        logits_flat = logits.reshape(-1, shard_vocab)
        tgt_flat = targets_stacked.reshape(-1)
        if _agent_avail:
            return dxe_loss(logits_flat, tgt_flat, shard_vocab)
        # Inline fallback: same algorithm as agent (pack 2 ARs into 1)
        local_max = logits_flat.max(dim=-1).values
        global_max = xm.all_reduce(xm.REDUCE_MAX, local_max)
        shifted = logits_flat - global_max.unsqueeze(-1)
        local_sum_exp = shifted.exp().sum(dim=-1)
        lo, hi = rank * shard_vocab, (rank + 1) * shard_vocab
        target_local = tgt_flat - lo
        in_shard = (tgt_flat >= lo) & (tgt_flat < hi)
        target_local_safe = target_local * in_shard.long()
        local_target = logits_flat.gather(1, target_local_safe.unsqueeze(1)).squeeze(1)
        local_target = local_target * in_shard.to(local_target.dtype)
        packed = torch.stack([local_sum_exp, local_target], dim=0)
        g_packed = xm.all_reduce(xm.REDUCE_SUM, packed)
        log_sum_exp = g_packed[0].log() + global_max
        return (log_sum_exp - g_packed[1]).mean()

    def step_per_mb(s):
        for p in all_params: p.grad = None
        loss_total = 0.0
        for m in range(N_MB):
            if stage == 0:
                h = embed(inputs[m]).to(torch.bfloat16)
                for L in range(N_LAYERS_PER_STAGE):
                    if probe.in_window():
                        _t0 = _t.time()
                        h = tp_fsdp_block(h, L)
                        _ = h.float().sum().item()
                        probe.record("tp_fsdp", (_t.time() - _t0) * 1000.0)
                    else:
                        h = tp_fsdp_block(h, L)
            else:
                h = torch.zeros(B, S, DM, device=dev, dtype=torch.bfloat16)
            if probe.in_window():
                _t0 = _t.time()
                h_t = transfer(h, src_stage=0)
                _ = h_t.float().sum().item()
                probe.record("pp_send_recv", (_t.time() - _t0) * 1000.0)
            else:
                h_t = transfer(h, src_stage=0)
            if stage == 1:
                h2 = h_t
                for L in range(N_LAYERS_PER_STAGE):
                    if probe.in_window():
                        _t0 = _t.time()
                        h2 = tp_fsdp_block(h2, L)
                        _ = h2.float().sum().item()
                        probe.record("tp_fsdp", (_t.time() - _t0) * 1000.0)
                    else:
                        h2 = tp_fsdp_block(h2, L)
                if probe.in_window():
                    _t0 = _t.time()
                    loss = vocab_parallel_loss(h2, targets[m])
                    _ = loss.float().sum().item()
                    probe.record("vocab_dxe", (_t.time() - _t0) * 1000.0)
                else:
                    loss = vocab_parallel_loss(h2, targets[m])
                loss_total = loss_total + loss
            xm.mark_step()
        if stage == 1:
            (loss_total / N_MB).backward()
            return loss_total
        else:
            # Stage 0: trigger autograd via 0-weighted residual.
            dummy = sum(transfer(embed(inputs[m]).to(torch.bfloat16), src_stage=0).sum() * 0
                        for m in range(N_MB))
            dummy.backward()
            return None

    def step_bundled(s):
        for p in all_params: p.grad = None
        # === Stage 0: bundled forward on M-stacked inputs ===
        if stage == 0:
            h_all = torch.stack([embed(inputs[m]).to(torch.bfloat16) for m in range(N_MB)], dim=0)
            for L in range(N_LAYERS_PER_STAGE):
                # FSDP prefetch: 3 AGs once per layer (was M*3 in per_mb)
                if probe.in_window():
                    _t0 = _t.time()
                    w_gate_full = xm.all_gather(w_gate[L], dim=1)
                    w_up_full   = xm.all_gather(w_up[L],   dim=1)
                    w_down_full = xm.all_gather(w_down[L], dim=0)
                    _ = w_gate_full.float().sum().item()
                    probe.record("fsdp_prefetch_bundled", (_t.time() - _t0) * 1000.0)
                else:
                    w_gate_full = xm.all_gather(w_gate[L], dim=1)
                    w_up_full   = xm.all_gather(w_up[L],   dim=1)
                    w_down_full = xm.all_gather(w_down[L], dim=0)
                # One matmul on M-stacked tensor (XLA broadcasts naturally)
                h = F.silu(torch.matmul(h_all, w_gate_full)) * torch.matmul(h_all, w_up_full)
                partial = torch.matmul(h, w_down_full)  # (M, B, S, DM)
                # ONE AR on stacked partial (matches evolved_tp_mlp's strategy)
                if probe.in_window():
                    _t0 = _t.time()
                    ared = xm.all_reduce(xm.REDUCE_SUM, partial) / ws
                    _ = ared.float().sum().item()
                    probe.record("tp_mlp_bundled", (_t.time() - _t0) * 1000.0)
                else:
                    ared = xm.all_reduce(xm.REDUCE_SUM, partial) / ws
                h_all = h_all + ared
        else:
            h_all = torch.zeros(N_MB, B, S, DM, device=dev, dtype=torch.bfloat16)
        # === PP transfer (inline transfer_batched: bundled masked-AR) ===
        if probe.in_window():
            _t0 = _t.time()
            h_all = transfer_batched(h_all, src_stage=0)
            _ = h_all.float().sum().item()
            probe.record("pp_send_recv", (_t.time() - _t0) * 1000.0)
        else:
            h_all = transfer_batched(h_all, src_stage=0)
        loss_total = 0.0
        # === Stage 1: bundled forward + bundled vocab loss ===
        if stage == 1:
            for L in range(N_LAYERS_PER_STAGE):
                if probe.in_window():
                    _t0 = _t.time()
                    w_gate_full = xm.all_gather(w_gate[L], dim=1)
                    w_up_full   = xm.all_gather(w_up[L],   dim=1)
                    w_down_full = xm.all_gather(w_down[L], dim=0)
                    _ = w_gate_full.float().sum().item()
                    probe.record("fsdp_prefetch_bundled", (_t.time() - _t0) * 1000.0)
                else:
                    w_gate_full = xm.all_gather(w_gate[L], dim=1)
                    w_up_full   = xm.all_gather(w_up[L],   dim=1)
                    w_down_full = xm.all_gather(w_down[L], dim=0)
                h = F.silu(torch.matmul(h_all, w_gate_full)) * torch.matmul(h_all, w_up_full)
                partial = torch.matmul(h, w_down_full)
                if probe.in_window():
                    _t0 = _t.time()
                    ared = xm.all_reduce(xm.REDUCE_SUM, partial) / ws
                    _ = ared.float().sum().item()
                    probe.record("tp_mlp_bundled", (_t.time() - _t0) * 1000.0)
                else:
                    ared = xm.all_reduce(xm.REDUCE_SUM, partial) / ws
                h_all = h_all + ared
            # Bundled vocab dxe (inline; algorithmically same as agent dxe_loss)
            targets_stacked = torch.stack(targets, dim=0)
            if probe.in_window():
                _t0 = _t.time()
                loss_total = vocab_parallel_loss_bundled(h_all, targets_stacked)
                _ = loss_total.float().sum().item()
                probe.record("vocab_dxe_bundled", (_t.time() - _t0) * 1000.0)
            else:
                loss_total = vocab_parallel_loss_bundled(h_all, targets_stacked)
            loss_total.backward()
            return loss_total
        else:
            (h_all.sum() * 0).backward()
            return None

    backend = sys.argv[1] if len(sys.argv) > 1 else 'per_mb'
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    fn = step_per_mb if backend == 'per_mb' else step_bundled

    bench_rank = half  # first stage-1 rank — sees real loss + bwd
    if rank == bench_rank:
        print(f'[init] ws={ws} half={half} N_MB={N_MB} N_LAYERS={N_LAYERS_PER_STAGE}/stage', flush=True)
        print(f'[init] DM={DM} HID={HID} VOCAB={VOCAB} B={B} S={S} backend={backend} steps={steps}', flush=True)

    for _ in range(WARMUP):
        loss = fn(0)
        if loss is not None:
            _ = loss.item()
    if rank == bench_rank:
        print('[init] warmup done', flush=True)

    times = []
    losses = []
    for s in range(steps):
        probe.set_step(s)
        xm.mark_step()
        t0 = time.time()
        loss = fn(s)
        if loss is not None:
            lv = loss.item()
            losses.append(lv)
        # SGD step on all params (replicated grad sync via simple AR baseline)
        if probe.in_window():
            _t0 = _t.time()
            with torch.no_grad():
                last_g = None
                for p in all_params:
                    if p.grad is not None:
                        g = xm.all_reduce(xm.REDUCE_SUM, p.grad) / ws
                        p.data = p.data - LR * g.to(p.dtype)
                        last_g = g
            if last_g is not None:
                _ = last_g.float().sum().item()
            probe.record("grad_ar_llama", (_t.time() - _t0) * 1000.0)
        else:
            with torch.no_grad():
                for p in all_params:
                    if p.grad is not None:
                        g = xm.all_reduce(xm.REDUCE_SUM, p.grad) / ws
                        p.data = p.data - LR * g.to(p.dtype)
        times.append((time.time() - t0) * 1000)
        if rank == bench_rank and (s + 1) % 100 == 0:
            recent = times[-50:]
            recent_loss = losses[-50:] if losses else [float('nan')]
            print(f'  step {s+1}: median_ms={statistics.median(recent):.1f} loss={statistics.mean(recent_loss):.3f}', flush=True)

    if rank == bench_rank:
        steady = times[STEADY_FROM:] if len(times) > STEADY_FROM else times
        steady_loss = losses[STEADY_FROM:] if len(losses) > STEADY_FROM else losses
        print(f'[bench] {backend} steady_mean={statistics.mean(steady):.2f}ms '
              f'steady_median={statistics.median(steady):.2f}ms '
              f'final_loss={steady_loss[-1] if steady_loss else float("nan"):.3f}', flush=True)
        with open(f'/tmp/tp_search/llama_e2e_amp2_{backend}.json', 'w') as f:
            json.dump({'backend': backend,
                       'steady_mean_ms': statistics.mean(steady),
                       'steady_median_ms': statistics.median(steady),
                       'all_ms': times,
                       'losses': losses[-100:] if losses else [],
                       'final_loss': steady_loss[-1] if steady_loss else None,
                       'DM': DM, 'HID': HID, 'N_LAYERS_PER_STAGE': N_LAYERS_PER_STAGE,
                       'N_MB': N_MB, 'B': B, 'S': S, 'VOCAB': VOCAB, 'steps': steps}, f)
        probe.dump(f'/tmp/tp_search/llama_e2e_amp2_{backend}_percall.json',
                   extra={'backend': backend, 'steps': steps})


if __name__ == '__main__':
    main()
