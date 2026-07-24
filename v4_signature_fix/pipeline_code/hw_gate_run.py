"""HW gate for kiss score_service (Approach 2').

Called as subprocess:
  torchrun --nproc_per_node=32 --standalone hw_gate_run.py \
    --problem PROBLEM --code-file /tmp/candidate.py

Exit code 0 = pass; 1 = fail. stderr contains failure details.
"""
import os, sys, argparse, importlib.util, traceback
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch.distributed as dist


def load_evolved_fn(code_file):
    spec = importlib.util.spec_from_file_location("cand", code_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in dir(mod):
        if name.startswith("evolved_"):
            return getattr(mod, name)
    return None


def check(problem, fn, rank, world, device):
    """Run candidate; return output tensor."""
    if problem == "leftpad_bcast":
        N, K = 64, 8
        ref = torch.cat([torch.zeros(K), torch.arange(N - K).float()])
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return fn(rank0, N, K, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "rangemask_bcast":
        N, K1, K2 = 64, 10, 30
        ar = torch.arange(N)
        ref = ((ar >= K1) & (ar < K2)).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return fn(rank0, N, K1, K2, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "stricttril_bcast":
        N = 32
        ref = (torch.arange(N).unsqueeze(1) > torch.arange(N).unsqueeze(0)).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "stackones_bcast":
        N, K = 32, 4
        ref = torch.zeros(K, N)
        for i in range(K): ref[i, :i+1] = 1.0
        rank0 = (ref if rank == 0 else torch.zeros(K, N)).to(device)
        return fn(rank0, N, K, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "onepos_bcast":
        N = 256
        ref = torch.zeros(N); ref[0] = 1.0
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "identity_matrix_bcast":
        N = 32
        ref = torch.eye(N)
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "diag_offset_bcast":
        N, K = 32, 3
        ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
        ref = (jj - ii == K).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return fn(rank0, N, K, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "col_mask_bcast":
        N, K = 32, 5
        jj = torch.arange(N).unsqueeze(0).expand(N, N)
        ref = (jj == K).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return fn(rank0, N, K, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "row_mask_bcast":
        N, K = 32, 7
        ii = torch.arange(N).unsqueeze(1).expand(N, N)
        ref = (ii == K).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return fn(rank0, N, K, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == 'magic_grid_bcast':
        N, K = 16, 5
        ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
        ref = (ii + jj + K).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return fn(rank0, N, K, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == 'row_id_grid_bcast':
        N = 32
        ref = (torch.arange(N).unsqueeze(1) + 1).expand(N, N).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == 'window_mask_bcast':
        N, W = 32, 3
        ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
        ref = ((ii - jj).abs() <= W).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return fn(rank0, N, W, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "col_id_grid_bcast":
        N = 32
        ref = torch.arange(N).unsqueeze(0).expand(N, N).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "scaled_arange_bcast":
        N = 128
        ref = (torch.arange(N) * 3).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "outer_prod_bcast":
        N = 16
        ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
        ref = (ii * jj).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "pair_max_bcast":
        N = 32
        ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
        ref = torch.max(ii, jj).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "abs_diff_bcast":
        N = 32
        ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
        ref = (ii - jj).abs().float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "clamp_range_bcast":
        N = 128; K1 = 10; K2 = 100
        ref = torch.clamp(torch.arange(N), K1, K2).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return fn(rank0, N, K1, K2, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "const_seven_bcast":
        N = 256
        ref = torch.full((N,), 7.0)
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "cumsum_ones_bcast":
        N = 256
        ref = torch.cumsum(torch.ones(N), dim=0).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "neg_range_bcast":
        N = 256
        ref = (torch.arange(N, dtype=torch.float32) * -1)
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "geometric_bcast":
        N = 16
        ref = torch.tensor([2.0 ** i for i in range(N)])
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "squared_arange_bcast":
        N = 64
        ref = torch.arange(N, dtype=torch.float32) * torch.arange(N, dtype=torch.float32)
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "arange_squeeze_bcast":
        N = 128
        ref = torch.arange(N, dtype=torch.float32).unsqueeze(0)
        rank0 = (ref if rank == 0 else torch.zeros(1, N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "checkerboard_bcast":
        N = 32
        ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
        ref = ((ii + jj) % 2).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "banded_upper_bcast":
        N, K1, K2 = 32, 2, 5
        ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
        ref = ((jj - ii >= K1) & (jj - ii < K2)).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return fn(rank0, N, K1, K2, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "first_k_bcast":
        N, K = 128, 32
        idx = torch.arange(N)
        ref = (idx < K).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return fn(rank0, N, K, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "mod_k_bcast":
        N, K = 64, 4
        idx = torch.arange(N)
        ref = (idx % K == 0).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return fn(rank0, N, K, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "lower_tri_bcast":
        N = 32
        ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
        ref = (ii >= jj).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "anti_diag_2d_bcast":
        N = 32
        ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
        ref = (ii + jj == N - 1).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "clamp_range_bcast":
        N, K1, K2 = 128, 10, 100
        ref = torch.clamp(torch.arange(N), K1, K2).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return fn(rank0, N, K1, K2, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "mod_sq_bcast":
        N, K = 32, 7
        idx = torch.arange(N)
        ref = ((idx * idx) % K).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return fn(rank0, N, K, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "xor_grid_bcast":
        N = 32
        ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
        ref = torch.bitwise_xor(ii, jj).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "popcount_bcast":
        N = 128
        idx = torch.arange(N)
        ref = torch.tensor([bin(int(i)).count("1") for i in idx]).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "triangle_num_bcast":
        N = 64
        idx = torch.arange(N)
        ref = (idx * (idx + 1) // 2).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "sign_alt_bcast":
        N = 32
        ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
        ref = ((-1.0) ** (ii + jj).float()).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "bimodal_dist_bcast":
        N = 128
        idx = torch.arange(N)
        ref = ((idx - N // 2) ** 2).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "gray_code_bcast":
        N = 128
        idx = torch.arange(N)
        ref = (idx ^ (idx >> 1)).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "compound_ij_bcast":
        N = 32
        ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
        mnj = torch.min(ii, jj); mxj = torch.max(ii, jj)
        ref = (mnj * mxj + (ii - jj) ** 2).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == "perm_shuffle_bcast":
        N = 128
        idx = torch.arange(N)
        ref = ((2 * idx) % N).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    raise ValueError(f"Unknown problem: {problem}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--code-file", required=True)
    args = ap.parse_args()

    dist.init_process_group("xla")
    rank = xr.global_ordinal()
    world = xr.world_size()
    device = xm.xla_device()

    fn = load_evolved_fn(args.code_file)
    if fn is None:
        print("NO_EVOLVED_FN", file=sys.stderr); sys.exit(11)

    try:
        out, expected = check(args.problem, fn, rank, world, device)
        xm.mark_step()
        out_cpu = out.cpu().float()
        exp_cpu = expected.float()
        if out_cpu.shape != exp_cpu.shape:
            print(f"SHAPE_MISMATCH rank={rank}: got {out_cpu.shape} expected {exp_cpu.shape}", file=sys.stderr)
            sys.exit(12)
        diff = (out_cpu - exp_cpu).abs().max().item()
        if diff > 1e-4:
            if rank == 0:
                num_wrong = int(((out_cpu - exp_cpu).abs() > 1e-4).sum().item())
                total = int(out_cpu.numel())
                pct_wrong = 100.0 * num_wrong / total
                print(f"VALUE_MISMATCH rank={rank}: {pct_wrong:.1f}%% of {total} elements wrong", file=sys.stderr)
            else:
                print(f"VALUE_MISMATCH rank={rank}", file=sys.stderr)
            sys.exit(13)
        print(f"HW_GATE_PASS rank={rank}", flush=True)
    except Exception:
        print("HW_GATE_EXCEPTION", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
