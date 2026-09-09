"""Real training probe: mini-transformer that uses the picked collective at every layer.
Configured by PROBLEM env var. 100 iters, 2-node, 64 rank."""
import os, sys, time, importlib.util
import torch, torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch.distributed as dist

PROBLEM = os.environ["PROBLEM"]
VARIANT = os.environ["VARIANT"]

# Find runtime file
paths = {
    "baseline": [
        f"/home/ubuntu/runtime_baseline/trainium_{PROBLEM}_baseline.py",
    ],
    "strat_pick": [
        f"/home/ubuntu/runtime/trainium_{PROBLEM}_2node.py",
    ],
        "kiss_pick": [
        f"/home/ubuntu/runtime_gated_leftpad_bcast_opus/trainium_leftpad_bcast_2node.py" if PROBLEM == "leftpad_bcast" else f"/home/ubuntu/runtime_gated_stricttril_bcast_opus/trainium_stricttril_bcast_2node.py" if PROBLEM == "stricttril_bcast" else f"/home/ubuntu/runtime_kiss_xor/trainium_{PROBLEM}_2node.py",
        f"/home/ubuntu/runtime_v3_{PROBLEM}/trainium_{PROBLEM}_2node.py",
        f"/home/ubuntu/runtime_gated_{PROBLEM}/trainium_{PROBLEM}_2node.py",
        f"/home/ubuntu/runtime_kiss_p75/trainium_{PROBLEM}_2node.py",
        f"/home/ubuntu/runtime_kiss_p71/trainium_{PROBLEM}_2node.py",
        f"/home/ubuntu/runtime_kiss_p70/trainium_{PROBLEM}_2node.py",
        f"/home/ubuntu/runtime_kiss_p67/trainium_{PROBLEM}_2node.py",
        f"/home/ubuntu/runtime_kiss_p61/trainium_{PROBLEM}_2node.py",
        f"/home/ubuntu/runtime_v4/trainium_{PROBLEM}_2node.py",
        f"/home/ubuntu/runtime_kiss_pick/kiss-sorcar_{PROBLEM}_2node.py",
        f"/home/ubuntu/runtime_v3/trainium_{PROBLEM}_2node.py",
        f"/home/ubuntu/runtime_v5/trainium_{PROBLEM}_2node.py",
        f"/home/ubuntu/runtime_v6/trainium_{PROBLEM}_2node.py",
        f"/home/ubuntu/runtime_v7/trainium_{PROBLEM}_2node.py",
        f"/home/ubuntu/runtime_v8/trainium_{PROBLEM}_2node.py",
        f"/home/ubuntu/runtime_v9/trainium_{PROBLEM}_2node.py",
        f"/home/ubuntu/runtime_vf/trainium_{PROBLEM}_2node.py",
        f"/home/ubuntu/runtime_ve/trainium_{PROBLEM}_2node.py",
        f"/home/ubuntu/runtime_vd/trainium_{PROBLEM}_2node.py",
        f"/home/ubuntu/runtime_vc/trainium_{PROBLEM}_2node.py",
        f"/home/ubuntu/runtime_vb/trainium_{PROBLEM}_2node.py",
        f"/home/ubuntu/runtime_va/trainium_{PROBLEM}_2node.py",
    ],
}[VARIANT]
RTFILE = None
for p in paths:
    if os.path.exists(p):
        RTFILE = p
        break
if RTFILE is None:
    print(f"MISSING_RUNTIME {PROBLEM} {VARIANT}", flush=True)
    sys.exit(1)

spec = importlib.util.spec_from_file_location("rt", RTFILE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
evolved_fn = None
for name in dir(mod):
    if name.startswith("evolved_"):
        evolved_fn = getattr(mod, name)
        break

dist.init_process_group("xla", init_method="xla://")
rank = xr.global_ordinal()
world = xr.world_size()
device = xm.xla_device()

BATCH, SEQ, DIM = 8, 128, 512
N_LAYERS = 4

# Set up aux input tensors per problem (matches problem semantics)
def setup_aux():
    """Return a dict of aux tensors that this problem's evolved_fn needs."""
    if PROBLEM == "dropout_mask_sync":
        aux = (torch.rand(BATCH, SEQ) > 0.1).float().to(device)
        rank0 = aux if rank == 0 else torch.zeros_like(aux)
        return {"input": rank0, "aux": aux, "call": lambda fn: fn(rank0, aux, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x * o.unsqueeze(-1)}
    if PROBLEM == "rope_bcast":
        aux = (torch.arange(SEQ * DIM).float().reshape(SEQ, DIM) / (SEQ * DIM)).to(device)
        rank0 = aux if rank == 0 else torch.zeros_like(aux)
        return {"call": lambda fn: fn(rank0, aux, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.unsqueeze(0) * 0.01}
    if PROBLEM == "rank_id_gather":
        my_id = torch.tensor([float(rank)]).to(device)
        return {"call": lambda fn: fn(my_id, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 0.001}
    if PROBLEM == "identity_bcast":
        N = 32
        rank0 = (torch.eye(N).float() if rank == 0 else torch.zeros(N, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o[:1, :1].sum() * 0.001}
    if PROBLEM == "tril_ones_bcast":
        N = 32
        ref = torch.tril(torch.ones(N, N)).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o[:1, :1].sum() * 0.001}
    if PROBLEM == "pos_encoding_bcast":
        N, D = 32, 16
        ref = (torch.arange(N).unsqueeze(1) * D + torch.arange(D).unsqueeze(0)).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, D)).to(device)
        return {"call": lambda fn: fn(rank0, N, D, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o[:1, :1].sum() * 1e-4}
    if PROBLEM == "inverse_perm_bcast":
        N = 32
        torch.manual_seed(rank)
        perm = torch.randperm(N).float().to(device)
        return {"call": lambda fn: fn(perm, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-5}
    if PROBLEM == "causal_mask_ar":
        L = 64
        ref = (torch.arange(L).unsqueeze(1) >= torch.arange(L).unsqueeze(0)).float()
        rank0 = (ref if rank == 0 else torch.zeros(L, L)).to(device)
        return {"call": lambda fn: fn(rank0, L, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o[:1, :1].sum() * 0.001}
    if PROBLEM == "global_arange_dispatch":
        N = 16
        local_slice = torch.arange(rank * N, (rank + 1) * N).float().to(device)
        return {"call": lambda fn: fn(local_slice, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-6}
    if PROBLEM == "strided_range_bcast":
        N, stride = 32, 3
        ref = torch.arange(0, N * stride, stride).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return {"call": lambda fn: fn(rank0, N, stride, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-5}
    if PROBLEM == "checkerboard_bcast":
        N = 32
        ref = ((torch.arange(N).unsqueeze(1) + torch.arange(N).unsqueeze(0)) % 2 == 0).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o[:1, :1].sum() * 0.001}
    if PROBLEM == "band_mask_bcast":
        N, K = 32, 3
        diff = (torch.arange(N).unsqueeze(1) - torch.arange(N).unsqueeze(0)).abs()
        ref = (diff <= K).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, K, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o[:1, :1].sum() * 0.001}
    if PROBLEM == "constval_bcast":
        N, val = 64, 3.14
        ref = torch.full((N,), val).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return {"call": lambda fn: fn(rank0, N, val, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-4}
    if PROBLEM == "cumsum_bcast":
        N = 64
        torch.manual_seed(0)
        xin = torch.randn(N).to(device)
        return {"call": lambda fn: fn(xin, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-4}
    if PROBLEM == "onehot_bcast":
        N, B = 32, 16
        torch.manual_seed(0)
        idx = torch.randint(0, N, (B,)).float().to(device)
        return {"call": lambda fn: fn(idx, N, B, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-5}
    if PROBLEM == "range_repeat_bcast":
        N, K = 8, 4
        ref = torch.arange(K).unsqueeze(0).expand(N, K).contiguous().float()
        rank0 = (ref if rank == 0 else torch.zeros(N, K)).to(device)
        return {"call": lambda fn: fn(rank0, N, K, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-4}

    if PROBLEM == "reverse_arange_bcast":
        N = 64
        ref = torch.arange(N - 1, -1, -1).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-6}
    if PROBLEM == "antidiag_mask_bcast":
        N = 32
        ref = (torch.arange(N).unsqueeze(1) + torch.arange(N).unsqueeze(0) == N - 1).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-5}
    if PROBLEM == "modidx_bcast":
        N, K = 64, 5
        ref = (torch.arange(N) % K).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return {"call": lambda fn: fn(rank0, N, K, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-6}
    if PROBLEM == "alternating_sign_bcast":
        N = 64
        ref = ((torch.arange(N) % 2) * -2 + 1).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-6}
    if PROBLEM == "block_arange_bcast":
        N, K = 8, 8
        ref = torch.arange(N * K).reshape(N, K).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, K)).to(device)
        return {"call": lambda fn: fn(rank0, N, K, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-6}

    if PROBLEM == "arange_scaled_bcast":
        N, val = 64, 2.5
        ref = (torch.arange(N).float() * val)
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return {"call": lambda fn: fn(rank0, N, val, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-6}
    if PROBLEM == "triu_ones_bcast":
        N = 32
        ref = torch.triu(torch.ones(N, N)).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-5}
    if PROBLEM == "arange_squared_bcast":
        N = 64
        ref = (torch.arange(N).float()) ** 2
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-6}
    if PROBLEM == "shifted_arange_bcast":
        N, offset = 64, 100
        ref = torch.arange(offset, offset + N).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return {"call": lambda fn: fn(rank0, N, offset, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-7}
    if PROBLEM == "diagvals_bcast":
        N = 32
        vals = torch.arange(1, N + 1).float()
        ref = torch.zeros(N, N)
        ref[torch.arange(N), torch.arange(N)] = vals
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-6}
    if PROBLEM == "block_eye_bcast":
        N, B = 8, 4
        ref = torch.zeros(N * B, N * B)
        for i in range(N):
            ref[i * B:(i + 1) * B, i * B:(i + 1) * B] = torch.eye(B)
        rank0 = (ref if rank == 0 else torch.zeros(N * B, N * B)).to(device)
        return {"call": lambda fn: fn(rank0, N, B, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-6}
    if PROBLEM == "chunk_sum_bcast":
        N, K = 8, 8
        ref = torch.tensor([sum(range(i * K, (i + 1) * K)) for i in range(N)]).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return {"call": lambda fn: fn(rank0, N, K, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-6}
    if PROBLEM == "triu_strict_bcast":
        N = 32
        ref = (torch.arange(N).unsqueeze(1) < torch.arange(N).unsqueeze(0)).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-5}
    if PROBLEM == "two_ar_batching":
        N = 64
        torch.manual_seed(0)
        ref = torch.randn(2, N).float()
        rank0 = (ref if rank == 0 else torch.zeros(2, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-5}
    if PROBLEM == "bool_eq_bcast":
        N, K = 64, 7
        ref = (torch.arange(N) == K).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return {"call": lambda fn: fn(rank0, N, K, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-4}


    if PROBLEM == "triangular_numbers_bcast":
        N = 64
        ref = torch.tensor([i * (i + 1) // 2 for i in range(N)]).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-8}
    if PROBLEM == "concat_arange_bcast":
        N = 32
        ref = torch.cat([torch.arange(N), torch.arange(N)]).float()
        rank0 = (ref if rank == 0 else torch.zeros(2 * N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-6}
    if PROBLEM == "repeat_arange_bcast":
        N, K = 16, 5
        ref = torch.arange(N).unsqueeze(0).expand(K, N).contiguous().float()
        rank0 = (ref if rank == 0 else torch.zeros(K, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, K, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-5}
    if PROBLEM == "outer_arange_bcast":
        N = 16
        a = torch.arange(N).float()
        ref = a.unsqueeze(1) * a.unsqueeze(0)
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-6}
    if PROBLEM == "range_sum_scalar_bcast":
        N = 32
        ref = torch.full((N,), (N * (N - 1)) // 2).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-6}
    if PROBLEM == "divby_mask_bcast":
        N, K = 64, 3
        ref = ((torch.arange(N) % K) == 0).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return {"call": lambda fn: fn(rank0, N, K, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-4}
    if PROBLEM == "linspace_bcast":
        N = 64
        hi = 10.0
        ref = torch.arange(N).float() * (hi / (N - 1))
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return {"call": lambda fn: fn(rank0, N, hi, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-6}
    if PROBLEM == "halfarange_bcast":
        N = 64
        ref = (torch.arange(N) // 2).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-6}
    if PROBLEM == "zerones_bcast":
        N = 32
        ref = torch.cat([torch.zeros(N), torch.ones(N)]).float()
        rank0 = (ref if rank == 0 else torch.zeros(2 * N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-4}

    if PROBLEM == "leftpad_bcast":
        N, K = 64, 8
        ref = torch.cat([torch.zeros(K), torch.arange(N - K).float()])
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return {"call": lambda fn: fn(rank0, N, K, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-6}
    if PROBLEM == "rangemask_bcast":
        N, K1, K2 = 64, 10, 30
        ar = torch.arange(N)
        ref = ((ar >= K1) & (ar < K2)).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return {"call": lambda fn: fn(rank0, N, K1, K2, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-4}
    if PROBLEM == "stricttril_bcast":
        N = 32
        ref = (torch.arange(N).unsqueeze(1) > torch.arange(N).unsqueeze(0)).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-5}
    if PROBLEM == "stackones_bcast":
        N, K = 32, 4
        ref = torch.zeros(K, N)
        for i in range(K):
            ref[i, :i+1] = 1.0
        rank0 = (ref if rank == 0 else torch.zeros(K, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, K, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-4}
    if PROBLEM == "onepos_bcast":
        N = 256
        ref = torch.zeros(N)
        ref[0] = 1.0
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-4}
    if PROBLEM == "identity_matrix_bcast":
        N = 32
        ref = torch.eye(N)
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-5}
    if PROBLEM == "diag_offset_bcast":
        N, K = 32, 3
        ii = torch.arange(N).unsqueeze(1)
        jj = torch.arange(N).unsqueeze(0)
        ref = (jj - ii == K).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, K, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-4}
    if PROBLEM == "col_mask_bcast":
        N, K = 32, 5
        jj = torch.arange(N).unsqueeze(0).expand(N, N)
        ref = (jj == K).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, K, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-4}
    if PROBLEM == "row_mask_bcast":
        N, K = 32, 7
        ii = torch.arange(N).unsqueeze(1).expand(N, N)
        ref = (ii == K).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, K, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-4}
    if PROBLEM == "row_id_grid_bcast":
        N = 32
        ref = (torch.arange(N).unsqueeze(1) + 1).expand(N, N).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-5}
    if PROBLEM == "magic_grid_bcast":
        N, K = 16, 5
        ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
        ref = (ii + jj + K).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, K, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-5}
    if PROBLEM == "window_mask_bcast":
        N, W = 32, 3
        ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
        ref = ((ii - jj).abs() <= W).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, W, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-5}
    if PROBLEM == "abs_diff_bcast":
        N = 32
        ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
        ref = (ii - jj).abs().float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-5}
    if PROBLEM == "hamming_dist_bcast":
        N = 16
        ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
        ref = torch.zeros(N, N)
        for i in range(N):
            for j in range(N):
                ref[i, j] = bin(i ^ j).count("1")
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o[:1, :1].sum() * 0.001}
    if PROBLEM == "nested_mod_bcast":
        N = 64
        idx = torch.arange(N)
        ref = ((idx * 3 + 1) % (idx % 7 + 2)).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-6}
    if PROBLEM == "sum_popcount_bcast":
        N = 32
        pc = torch.tensor([bin(int(i)).count("1") for i in range(N)]).float()
        ref = (pc.unsqueeze(1) + pc.unsqueeze(0))
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o[:1, :1].sum() * 0.001}
    if PROBLEM == "piecewise_bcast":
        N = 64
        idx = torch.arange(N)
        ref = torch.where(idx < N // 2, idx * idx, (N - idx) * (N - idx)).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-8}
    if PROBLEM == "cond_xor_bcast":
        N = 16
        ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
        ref = torch.where((ii + jj) % 2 == 0, torch.bitwise_xor(ii, jj), torch.zeros_like(ii)).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o[:1, :1].sum() * 0.001}

    if PROBLEM == "xor_grid_bcast":
        N = 32
        ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
        ref = torch.bitwise_xor(ii, jj).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-6}
    raise ValueError(f"Unknown problem {PROBLEM}")

setup = setup_aux()

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(DIM, DIM * 4)
        self.down = nn.Linear(DIM * 4, DIM)
        self.ln = nn.LayerNorm(DIM)
    def forward(self, x):
        # Call the picked collective; use its output as a small perturbation
        out = setup["call"](evolved_fn)
        h = self.ln(x)
        h = setup["apply"](h, out)  # applies problem-specific perturbation
        h = self.up(h); h = torch.relu(h); h = self.down(h)
        return x + h

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(1000, DIM)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.head = nn.Linear(DIM, 1000)
    def forward(self, tokens):
        x = self.emb(tokens)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)

model = Model().to(device)
optim = torch.optim.SGD(model.parameters(), lr=1e-2)

torch.manual_seed(42)
tokens = torch.randint(0, 1000, (BATCH, SEQ)).to(device)
targets = torch.randint(0, 1000, (BATCH, SEQ)).to(device)
loss_fn = nn.CrossEntropyLoss()

# Warmup
for _ in range(3):
    optim.zero_grad()
    logits = model(tokens)
    loss = loss_fn(logits.reshape(-1, 1000), targets.reshape(-1))
    loss.backward()
    optim.step()
    xm.mark_step()
xm.wait_device_ops()
xm.rendezvous("pre_bench")

# Real training
N_ITER = 100
losses = []
t0 = time.time()
for i in range(N_ITER):
    optim.zero_grad()
    logits = model(tokens)
    loss = loss_fn(logits.reshape(-1, 1000), targets.reshape(-1))
    loss.backward()
    optim.step()
    xm.mark_step()
    losses.append(loss.item())
xm.wait_device_ops()
t1 = time.time()

if rank == 0:
    print(f"REAL_PROBLEM {PROBLEM}", flush=True)
    print(f"REAL_VARIANT {VARIANT}", flush=True)
    print(f"REAL_MS_PER_ITER {(t1-t0)/N_ITER*1000:.4f}", flush=True)
    print(f"REAL_LOSS_INITIAL {losses[0]:.6f}", flush=True)
    print(f"REAL_LOSS_FINAL {losses[-1]:.6f}", flush=True)
    print(f"REAL_LOSS_DESCENT {losses[0] - losses[-1]:.6f}", flush=True)
