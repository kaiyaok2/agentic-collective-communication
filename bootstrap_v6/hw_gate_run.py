"""HW gate runner. Called under torchrun with --nproc_per_node=32.
Usage:
  torchrun --nproc_per_node=32 --nnodes=2 --node_rank=X --master_addr=... hw_gate_run.py --problem P --code-file /tmp/c.py

Loads candidate, runs it on the XLA device, compares against reference computed on CPU.
"""
import os, sys, argparse, importlib.util, traceback
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch.distributed as dist


def load_evolved_fn(code_file):
    spec = importlib.util.spec_from_file_location('cand', code_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in dir(mod):
        if name.startswith('evolved_'):
            return getattr(mod, name)
    return None


def check(problem, fn, rank, world, device):
    if problem == 'xor_grid_bcast':
        N = 32
        ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
        ref = torch.bitwise_xor(ii, jj).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == 'gray_code_bcast':
        N = 128
        idx = torch.arange(N)
        ref = (idx ^ (idx >> 1)).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == 'piecewise_bcast':
        N = 64
        idx = torch.arange(N)
        ref = torch.where(idx < N//2, idx*idx, (N-idx)*(N-idx)).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == 'triangle_num_bcast':
        N = 64
        idx = torch.arange(N)
        ref = (idx * (idx + 1) // 2).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == 'popcount_bcast':
        N = 128
        idx = torch.arange(N)
        ref = torch.tensor([bin(int(i)).count('1') for i in idx]).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == 'hamming_dist_bcast':
        N = 16
        ref = torch.zeros(N, N)
        for i in range(N):
            for j in range(N):
                ref[i, j] = bin(i ^ j).count('1')
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == 'cond_xor_bcast':
        N = 16
        ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
        ref = torch.where((ii + jj) % 2 == 0, torch.bitwise_xor(ii, jj), torch.zeros_like(ii)).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == 'sum_popcount_bcast':
        N = 32
        pc = torch.tensor([bin(int(i)).count('1') for i in range(N)]).float()
        ref = pc.unsqueeze(1) + pc.unsqueeze(0)
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == 'sign_alt_bcast':
        N = 32
        ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
        ref = ((-1.0) ** (ii + jj).float()).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == 'perm_shuffle_bcast':
        N = 128
        idx = torch.arange(N)
        ref = ((2 * idx) % N).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == 'mod_sq_bcast':
        N, K = 32, 7
        idx = torch.arange(N)
        ref = ((idx * idx) % K).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return fn(rank0, N, K, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    if problem == 'nested_mod_bcast':
        N = 64
        idx = torch.arange(N)
        ref = ((idx * 3 + 1) % (idx % 7 + 2)).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
    raise ValueError('Unknown problem: ' + problem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--problem', required=True)
    ap.add_argument('--code-file', required=True)
    args = ap.parse_args()

    dist.init_process_group('xla')
    rank = xr.global_ordinal()
    world = xr.world_size()
    device = xm.xla_device()

    fn = load_evolved_fn(args.code_file)
    if fn is None:
        print('NO_EVOLVED_FN', file=sys.stderr); sys.exit(11)
    try:
        out, expected = check(args.problem, fn, rank, world, device)
        xm.mark_step()
        out_cpu = out.cpu().float()
        exp_cpu = expected.float()
        if out_cpu.shape != exp_cpu.shape:
            print('SHAPE_MISMATCH rank=' + str(rank) + ': got ' + str(out_cpu.shape) + ' expected ' + str(exp_cpu.shape), file=sys.stderr)
            sys.exit(12)
        diff = (out_cpu - exp_cpu).abs().max().item()
        if diff > 1e-4:
            if rank == 0:
                num_wrong = int(((out_cpu - exp_cpu).abs() > 1e-4).sum().item())
                total = int(out_cpu.numel())
                pct_wrong = 100.0 * num_wrong / total
                print('VALUE_MISMATCH rank=' + str(rank) + ': ' + str(round(pct_wrong,1)) + ' pct of ' + str(total) + ' elements wrong', file=sys.stderr)
            else:
                print('VALUE_MISMATCH rank=' + str(rank), file=sys.stderr)
            sys.exit(13)
        print('HW_GATE_PASS rank=' + str(rank), flush=True)
    except Exception:
        print('HW_GATE_EXCEPTION', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
