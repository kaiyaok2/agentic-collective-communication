"""Simpler RT probe: RUNTIME_FILE + PROBLEM env vars; forward the setup from real_training_multi setup_aux."""
import os, sys, time, importlib.util
import torch, torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch.distributed as dist

# Delegate to the real_training_multi setup_aux via import
RTM_PATH = "/home/ubuntu/agentic-collective-communication/v4_signature_fix/pipeline_code/real_training_multi.py"

PROBLEM = os.environ["PROBLEM"]
RUNTIME_FILE = os.environ["RUNTIME_FILE"]

dist.init_process_group("xla")
rank = xr.global_ordinal()
world = xr.world_size()
device = xm.xla_device()

# Import evolved fn from RUNTIME_FILE
spec = importlib.util.spec_from_file_location("rt", RUNTIME_FILE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
evolved_fn = None
for name in dir(mod):
    if name.startswith("evolved_"):
        evolved_fn = getattr(mod, name)
        break

# Recreate setup_aux logic inline for the problems we care about
def setup_aux():
    if PROBLEM == "hamming_dist_bcast":
        N = 16
        ref = torch.zeros(N, N)
        for i in range(N):
            for j in range(N):
                ref[i, j] = bin(i ^ j).count("1")
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o[:1, :1].sum() * 0.001}
    if PROBLEM == "xor_grid_bcast":
        N = 32
        ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
        ref = torch.bitwise_xor(ii, jj).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o[:1, :1].sum() * 0.001}
    if PROBLEM == "sum_popcount_bcast":
        N = 32
        pc = torch.tensor([bin(int(i)).count("1") for i in range(N)]).float()
        ref = (pc.unsqueeze(1) + pc.unsqueeze(0))
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o[:1, :1].sum() * 0.001}
    if PROBLEM == "gray_code_bcast":
        N = 128
        idx = torch.arange(N)
        ref = (idx ^ (idx >> 1)).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-6}
    if PROBLEM == "popcount_bcast":
        N = 128
        idx = torch.arange(N)
        ref = torch.tensor([bin(int(i)).count("1") for i in idx]).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-6}
    if PROBLEM == "sign_alt_bcast":
        N = 32
        ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
        ref = ((-1.0) ** (ii + jj).float()).float()
        rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o[:1, :1].sum() * 0.001}
    if PROBLEM == "perm_shuffle_bcast":
        N = 128
        idx = torch.arange(N)
        ref = ((2 * idx) % N).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-6}
    if PROBLEM == "mod_sq_bcast":
        N, K = 32, 7
        idx = torch.arange(N)
        ref = ((idx * idx) % K).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return {"call": lambda fn: fn(rank0, N, K, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-6}
    if PROBLEM == "triangle_num_bcast":
        N = 64
        idx = torch.arange(N)
        ref = (idx * (idx + 1) // 2).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-8}
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
    if PROBLEM == "nested_mod_bcast":
        N = 64
        idx = torch.arange(N)
        ref = ((idx * 3 + 1) % (idx % 7 + 2)).float()
        rank0 = (ref if rank == 0 else torch.zeros(N)).to(device)
        return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2), "apply": lambda x, o: x + o.sum() * 1e-6}
    raise ValueError(f"Unknown problem {PROBLEM}")

setup = setup_aux()
DIM = 128
NLAYERS = 2

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(DIM, DIM * 4)
        self.down = nn.Linear(DIM * 4, DIM)
        self.ln = nn.LayerNorm(DIM)
    def forward(self, x):
        out = setup["call"](evolved_fn)
        h = self.ln(x)
        h = setup["apply"](h, out)
        h = self.up(h); h = torch.relu(h); h = self.down(h)
        return x + h

torch.manual_seed(0)
model = nn.Sequential(*[Block() for _ in range(NLAYERS)]).to(device)
opt = torch.optim.SGD(model.parameters(), lr=1e-4)

N_ITERS = int(os.environ.get("N_ITERS", "100"))
# warmup
x = torch.randn(4, DIM).to(device)
for _ in range(3):
    y = model(x)
    loss = y.pow(2).mean()
    loss.backward()
    opt.step(); opt.zero_grad()
    xm.mark_step()

# timed
torch.manual_seed(1)
x = torch.randn(4, DIM).to(device)
t0 = time.perf_counter()
for _ in range(N_ITERS):
    y = model(x)
    loss = y.pow(2).mean()
    loss.backward()
    opt.step(); opt.zero_grad()
    xm.mark_step()
xm.mark_step()
xm.wait_device_ops()
t1 = time.perf_counter()
if rank == 0:
    print(f"RT_TIME_MS_PER_ITER {1000.0 * (t1-t0) / N_ITERS:.4f}", flush=True)
