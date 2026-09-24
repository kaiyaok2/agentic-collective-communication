"""Warm-cache RT probe for the fair-divergence family problems.

Unlike rt_run_v12.py (whose setup_aux hardcodes the old *_bcast/*_chal catalog),
this loader pulls the problem definition from the repo registry, so ANY registered
diverge problem (r59b_*, r60*, r61*, r63*, r65*, r66*, r67*, r68*, r69*, ...) can be
timed with no per-problem code here.

Env:
  PROBLEM       registered problem name (e.g. r68_bidi_b03_d8_p1024)
  RUNTIME_FILE  path to the candidate code (staged sorcar.py / overlay.py)
  N_ITERS       timed iterations (default 50); NUM_NODES (default 7)

Config matches the campaign sim exactly: world_size=224, num_devices=112,
cores_per_device=2, num_nodes=7 (7x trn1.32xlarge = 224 ranks). Launched via
torchrun; emits `RT_TIME_MS_PER_ITER <ms>` on rank 0.
"""
import glob
import importlib
import importlib.util
import os
import time

import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch.distributed as dist

ACC = os.environ.get("ACC_REPO", "/home/ubuntu/agentic-collective-communication")
PROBLEM = os.environ["PROBLEM"]
RUNTIME_FILE = os.environ["RUNTIME_FILE"]
N_ITERS = int(os.environ.get("N_ITERS", "50"))
NUM_NODES = int(os.environ.get("NUM_NODES", "7"))
NUM_DEVICES = int(os.environ.get("NUM_DEVICES", "112"))
CORES_PER_DEVICE = int(os.environ.get("CORES_PER_DEVICE", "2"))

dist.init_process_group("xla")
rank = xr.global_ordinal()
world = xr.world_size()
device = xm.xla_device()

# --- register every diverge problem, then fetch this one from the registry ---
import sys
if ACC not in sys.path:
    sys.path.insert(0, ACC)
from search.problems import get_problem  # noqa: E402

for mp in sorted(glob.glob(os.path.join(ACC, "search", "problems_diverge_*.py"))):
    modname = "search." + os.path.splitext(os.path.basename(mp))[0]
    try:
        importlib.import_module(modname)  # module-level register_all() self-registers
    except Exception as e:  # noqa: BLE001
        if rank == 0:
            print(f"[rt_diverge] warn: import {modname} failed: {e}", flush=True)

prob = get_problem(PROBLEM)

# Build THIS rank's input EXACTLY as the fp32 gate does: generate_test_case(world).
# (The old world*part reconstruction assumed a shard-per-rank payload and fed a
#  wrong-shaped, ~112x oversized tensor to FIXED-payload problems -> broadcast/view
#  crash on both pipelines. Using the gate's own per_rank_args makes RT == gate.)
torch.manual_seed(1234)
_tc = prob.generate_test_case(world)
_pra = _tc["per_rank_args"]
assert rank < len(_pra), f"per_rank_args has {len(_pra)} entries < rank {rank} @ world {world}"
x_in = _pra[rank]["x"].detach().clone().to(device)
del _tc, _pra
N = x_in.numel()
part = N  # per-rank payload (scales with world or fixed, per the problem)

# --- load the candidate function (named {PROBLEM}_fn per the forced signature) ---
spec = importlib.util.spec_from_file_location("rt_candidate", RUNTIME_FILE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fn_name = getattr(prob, "evolved_fn_name", f"{PROBLEM}_fn")
evolved_fn = getattr(mod, fn_name, None)
if evolved_fn is None:  # fall back to first *_fn / evolved_* symbol
    for nm in dir(mod):
        if nm.endswith("_fn") or nm.startswith("evolved_"):
            evolved_fn = getattr(mod, nm)
            break
assert callable(evolved_fn), f"no candidate fn in {RUNTIME_FILE}"


def call(fn):
    return prob.call_candidate(fn, {"x": x_in}, {}, rank, world,
                               NUM_DEVICES, CORES_PER_DEVICE, xm, torch,
                               num_nodes=NUM_NODES)


# embed the collective in a small trained graph so XLA can't DCE the all_reduce
DIM = 128
NLAYERS = 2


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(DIM, DIM * 4)
        self.down = nn.Linear(DIM * 4, DIM)
        self.ln = nn.LayerNorm(DIM)

    def forward(self, x):
        out = call(evolved_fn)
        h = self.ln(x)
        h = h + out.sum() * 1e-12  # consume the collective output
        h = self.up(h); h = torch.relu(h); h = self.down(h)
        return x + h


torch.manual_seed(0)
model = nn.Sequential(*[Block() for _ in range(NLAYERS)]).to(device)
opt = torch.optim.SGD(model.parameters(), lr=1e-4)

# warmup (compiles + warms the Neuron cache)
x = torch.randn(4, DIM).to(device)
for _ in range(5):
    y = model(x)
    loss = y.pow(2).mean()
    loss.backward()
    opt.step(); opt.zero_grad()
    xm.mark_step()
xm.wait_device_ops()

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
    print(f"RT_TIME_MS_PER_ITER {1000.0 * (t1 - t0) / N_ITERS:.4f}", flush=True)
    print(f"RT_META problem={PROBLEM} part={part} N={N} world={world} "
          f"file={os.path.basename(os.path.dirname(RUNTIME_FILE))}/{os.path.basename(RUNTIME_FILE)}",
          flush=True)
