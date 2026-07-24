# Reproduction Guide — V4 Signature-Fix Pipeline (2026-07-24)

Complete step-by-step to reproduce every claim in `FULL_SCOREBOARD.md`. Builds on the CB v3 pipeline (`v3-final-pipeline-2026-07-23`) but with the critical prompt-population fix that changes kiss's behavior.

## 0. Prerequisites

- AWS account with:
  - Capacity Block reservation for 2× trn1.32xlarge (us-west-2, adjacent AZ). CB IDs used: `cr-042a5e2afd89bfc6b` (v3), `cr-00838c418d66f6883` (v4).
  - Amazon Bedrock model access in us-east-1 for at least: `us.anthropic.claude-opus-4-8`, `us.anthropic.claude-sonnet-5`. (Optional: `us.anthropic.claude-fable-5`, `us.anthropic.claude-haiku-4-5-20251001-v1:0`.)
  - Note: `claude-sonnet-4-6` requires Marketplace subscription that our IAM role doesn't have — not usable without that subscription.
  - IAM role `OverlayCCL-Kaiyao-KissWorker` (or equivalent) with EC2 + S3 + Bedrock InvokeModel permissions.
- Security group `overlayccl-usw2` with self-referencing all-traffic egress rule (see [efa-peer-self-root-cause] memory — missing = cross-node hangs).
- SSH key pair `Kaiyao` in us-west-2.

## 1. Launch 2 nodes against your CB

```bash
# NIC spec for 8-EFA-per-node
cat > nic_spec.json <<'EOF'
[
  {"DeviceIndex":0,"NetworkCardIndex":0,"InterfaceType":"efa","Groups":["<sg-id>"],"SubnetId":"<subnet-id>"},
  {"DeviceIndex":1,"NetworkCardIndex":1,"InterfaceType":"efa","Groups":["<sg-id>"],"SubnetId":"<subnet-id>"},
  {"DeviceIndex":1,"NetworkCardIndex":2,"InterfaceType":"efa","Groups":["<sg-id>"],"SubnetId":"<subnet-id>"},
  {"DeviceIndex":1,"NetworkCardIndex":3,"InterfaceType":"efa","Groups":["<sg-id>"],"SubnetId":"<subnet-id>"},
  {"DeviceIndex":1,"NetworkCardIndex":4,"InterfaceType":"efa","Groups":["<sg-id>"],"SubnetId":"<subnet-id>"},
  {"DeviceIndex":1,"NetworkCardIndex":5,"InterfaceType":"efa","Groups":["<sg-id>"],"SubnetId":"<subnet-id>"},
  {"DeviceIndex":1,"NetworkCardIndex":6,"InterfaceType":"efa","Groups":["<sg-id>"],"SubnetId":"<subnet-id>"},
  {"DeviceIndex":1,"NetworkCardIndex":7,"InterfaceType":"efa","Groups":["<sg-id>"],"SubnetId":"<subnet-id>"}
]
EOF

aws ec2 run-instances \
  --instance-type trn1.32xlarge \
  --image-id ami-0b4b7dd2217eda98d \
  --key-name Kaiyao \
  --count 2 \
  --iam-instance-profile Name=OverlayCCL-Kaiyao-KissWorker \
  --instance-market-options MarketType=capacity-block \
  --capacity-reservation-specification CapacityReservationTarget={CapacityReservationId=<your-CB>} \
  --network-interfaces file://nic_spec.json \
  --user-data file://cb_v4_userdata.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=cb-v4-kiss}]' \
  --region us-west-2
```

The `cb_v4_userdata.sh` (see v3-final-pipeline branch) installs Neuron SDK 2.24, creates the Python venv, and fetches `state_snapshot.tar.gz` from S3.

Attach public EIPs to both instances' primary NICs:
```bash
aws ec2 associate-address --allocation-id <eip-alloc> --network-interface-id <eni-0-of-master>
aws ec2 associate-address --allocation-id <eip-alloc-2> --network-interface-id <eni-0-of-worker>
```

## 2. Post-bootstrap fixes (one-time, both nodes)

The `cb_v4_userdata.sh` snapshot script installs pieces but has 3 known issues:

**Issue 1: python3.13 for kiss venv.** The snapshot may fail on python3.13 install (missing PPA). Fix:
```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get install -y python3.13 python3.13-venv python3.13-dev
rm -rf /home/ubuntu/kiss/.venv
python3.13 -m venv /home/ubuntu/kiss/.venv
/home/ubuntu/kiss/.venv/bin/pip install --upgrade pip
/home/ubuntu/kiss/.venv/bin/pip install -e /home/ubuntu/kiss
/home/ubuntu/kiss/.venv/bin/pip install boto3 botocore
```

**Issue 2: `cb2_verify` path.** The bootstrap creates `/home/ubuntu/cb2_verify/repo` as a plain dir; the newer test repo is at `/home/ubuntu/cb2_verify_v3/repo`. Fix:
```bash
mv /home/ubuntu/cb2_verify /home/ubuntu/cb2_verify_patches
mkdir /home/ubuntu/cb2_verify
ln -s /home/ubuntu/cb2_verify_v3/repo /home/ubuntu/cb2_verify/repo
```

**Issue 3: patched files.** Overlay the following on top of the v3 repo:
```bash
cp cb2_verify_patches/repo/prompts/generic_evolution.md cb2_verify_v3/repo/prompts/
cp cb2_verify_patches/repo/experiments/ablation_kiss_vs_cc/score_service_v2.py cb2_verify_v3/repo/experiments/ablation_kiss_vs_cc/
cp cb2_verify_patches/repo/search/correctness_test.py cb2_verify_v3/repo/search/
cp cb2_verify_patches/repo/search/strategy_enumerate_phase3.py cb2_verify_v3/repo/search/
cp cb2_verify_patches/repo/experiments/ablation_kiss_vs_cc/kiss_phase3.py cb2_verify_v3/repo/experiments/ablation_kiss_vs_cc/
# Also copy the extended problem catalog (with row_id_grid_bcast etc.)
cp cb2_verify_patches/repo/search/problems_kiss_verify.py cb2_verify_v3/repo/search/
```

## 3. THE CRITICAL FIX — signature_doc population

Without this fix, kiss's `{signature_doc}` placeholder is never replaced. Kiss sees the literal string `{signature_doc}` in its prompt and has to infer the formula from the function name alone. WITH this fix, kiss sees the actual `Formula: ...` docstring.

In `kiss_phase3.py`:

```python
def get_baseline_code(problem_name):
    r = subprocess.run(
        [NEURON_PY, "-c",
         "import sys, json; sys.path.insert(0, "
         f"{ACC!r});"
         # ADD problem catalog imports:
         "from search.problems import get_problem; "
         "import search.problems_kiss_verify; "
         "import search.problems_modext; "
         "import search.problems_novel_v4; "
         "import search.problems_novel_v5;"
         f"p = get_problem('{problem_name}');"
         "tmpls = p.builtin_templates;"
         "k = next(iter(tmpls.keys()));"
         # ADD signature + signature_doc:
         "print(json.dumps({'name': k, 'code': tmpls[k], "
         "'signature': p.signature, 'signature_doc': p.signature_doc}))"],
        capture_output=True, text=True, timeout=30)
    ...

def main():
    ...
    prompt = prompt.replace("{current_code}", base["code"])
    # ADD:
    prompt = prompt.replace("{signature}", base.get("signature", ""))
    prompt = prompt.replace("{signature_doc}", base.get("signature_doc", ""))
    prompt = prompt.replace("{current_sim_time}", str(base_score.get("sim_time_us", 0)))
    ...
```

The exact patched file is in `pipeline_code/kiss_phase3.py`.

Also patch `score_service_v2.py` and `run_search.py` to import the problem catalogs so both kiss and strat can look up novel problems:
```bash
# In score_service_v2.py after "from search.problems import get_problem":
#     import search.problems_kiss_verify
#     import search.problems_novel_v4
#     import search.problems_novel_v5
#     import search.problems_modext
```

Update the Bedrock model shim (`kiss_token_shim.py`) with current model IDs:
```python
_MODEL_ID_TO_BEDROCK = {
    "claude-opus-4-8": "us.anthropic.claude-opus-4-8",
    "claude-sonnet-5": "us.anthropic.claude-sonnet-5",
    "claude-fable-5": "us.anthropic.claude-fable-5",
    "claude-haiku-4-5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    # kiss's default model name still refers to sonnet-4-5, map it to opus-4-8
    "claude-sonnet-4-5-20250929": "us.anthropic.claude-opus-4-8",
    "claude-sonnet-4-5": "us.anthropic.claude-opus-4-8",
    ...
}
```

## 4. Environment variables (every session)

```bash
export PATH=/home/ubuntu/venv/bin:$PATH
export LD_LIBRARY_PATH=/opt/aws/neuron/lib
export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1
export USE_BEDROCK=1 AWS_REGION=us-east-1 BEDROCK_REGION=us-east-1
export NEURON_PY=/home/ubuntu/venv/bin/python
export MASTER_IP=172.31.17.9      # your master private IP
export WORKER_IP=172.31.26.232    # your worker private IP
export HW_GATE=1
export HW_GATE_SCRIPT=/home/ubuntu/hw_gate_run.py
export KISS_MODEL=claude-opus-4-8  # or claude-sonnet-5, claude-fable-5
unset ANTHROPIC_API_KEY
```

`KISS_MODEL` env var replaces the hardcoded model in `kiss_phase3.py`. Patch:
```python
agent.run(model_name=os.environ.get("KISS_MODEL", "claude-opus-4-8"), ...)
```

## 5. Adding novel problems (P_87-P_95)

The 9 novel problems used in this study are in:
- `pipeline_code/problems_novel_v4.py` — 6 problems (mod_sq, xor_grid, popcount, triangle_num, sign_alt, bimodal_dist)
- `pipeline_code/problems_novel_v5.py` — 3 problems (gray_code, compound_ij, perm_shuffle)

Place these files at `/home/ubuntu/cb2_verify/repo/search/` on both master and worker.

For each problem, also add its HW-gate check to `hw_gate_run.py` (see `pipeline_code/hw_gate_run.py` for the full patched version):
```python
if problem == "xor_grid_bcast":
    N = 32
    ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
    ref = torch.bitwise_xor(ii, jj).float()
    rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
    return fn(rank0, N, rank, world, world//2, 2, xm, torch, num_nodes=1), ref
```

## 6. Kiss search on one problem

```bash
export KISS_MODEL=claude-opus-4-8

rm -rf /tmp/run_xor
mkdir -p /tmp/run_xor
timeout 900 /home/ubuntu/kiss/.venv/bin/python \
  /home/ubuntu/cb2_verify/repo/experiments/ablation_kiss_vs_cc/kiss_phase3.py \
  --problem xor_grid_bcast --pattern moe --output-dir /tmp/run_xor \
  --max-budget 12.0 --max-steps 40 --num-nodes 2 > /tmp/run_xor/run.log 2>&1

cat /tmp/run_xor/kiss_summary.json
```

`kiss_summary.json` fields: `best_sim_time_us`, `best_name`, `n_score_calls`, `wall_seconds`.

## 7. Strat search on same problem

```bash
export STRAT_HW_GATE=1

rm -rf /tmp/strat_xor
mkdir -p /tmp/strat_xor
timeout 600 /home/ubuntu/venv/bin/python \
  /home/ubuntu/cb2_verify/repo/experiments/run_search.py \
  --problem xor_grid_bcast --pattern moe \
  --phase3-style strategy-enumerate --llm-model opus \
  --num-nodes 2 --worker-addrs $WORKER_IP --master-addr $MASTER_IP \
  --output-dir /tmp/strat_xor > /tmp/strat_xor.log 2>&1

grep -E "Winner|SimTime|HW_GATE_FAIL" /tmp/strat_xor.log
```

Strat writes runtime to `/home/ubuntu/runtime/trainium_${P}_2node.py` — copy it before the next problem's run overwrites it.

## 8. HW gate (no-leak diagnostics)

`hw_gate_2node.sh` calls `hw_gate_run.py` on both nodes via torchrun. Diagnostics returned to kiss:
- `HW_GATE_PASS` (empty stdout → sim_time is trusted)
- `SHAPE_MISMATCH: got=(a,b) expected=(c,d)` — output shape differs (no leak: shapes are structural)
- `VALUE_MISMATCH rank=0: X% of Y elements wrong` — coarse percentage, no numeric values or indices
- `HW_GATE_EXCEPTION` — code crashed compiling or executing at 64-rank

**Sanitization**: `pipeline_code/hw_gate_run.py` was patched to remove:
- `max_abs_diff={diff:.6f}` — leaked the exact numerical error magnitude
- `at index {worst_flat}` — leaked the position of the worst error

If you skip this sanitization, kiss can probe with `return torch.zeros_like(x)` to read back `max_abs_diff = <max reference value>` and reverse-engineer the reference. This IS reward-hacking. All results in the scoreboard were run WITH sanitization on.

## 9. Real training on a kiss-winner

Extract kiss's best code from `candidates.jsonl`:
```python
import json
best_sim, best_code = 1e18, None
for l in open("/tmp/run_xor/candidates.jsonl"):
    d = json.loads(l)
    r = d.get("result", {})
    if r.get("ok") and r.get("sim_time_us", 1e18) < best_sim:
        best_sim = r["sim_time_us"]
        best_code = d["code"]

header = "import torch\nimport torch_xla.core.xla_model as xm\nimport torch_xla.runtime as xr\n\n"
with open("/home/ubuntu/runtime_kiss_xor/trainium_xor_grid_bcast_2node.py", "w") as f:
    f.write(header + best_code)
```

Also create baseline:
```python
# /home/ubuntu/runtime_baseline/trainium_xor_grid_bcast_baseline.py
def evolved_p88(o, N, rank, world_size, num_devices,
                cores_per_device, xm, torch, num_nodes=1):
    return xm.all_reduce(xm.REDUCE_SUM, o)
```

Add setup_aux entry to `real_training_multi.py`:
```python
if PROBLEM == "xor_grid_bcast":
    N = 32
    ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
    ref = torch.bitwise_xor(ii, jj).float()
    rank0 = (ref if rank == 0 else torch.zeros(N, N)).to(device)
    return {"call": lambda fn: fn(rank0, N, rank, world, 32, 2, xm, torch, num_nodes=2),
            "apply": lambda x, o: x + o.sum() * 1e-6}
```

Run 2-node 64-rank training on both master and worker (see `pipeline_code/rt_xor.sh` for the full driver script including zombie cleanup between runs — critical for repeated torchrun invocations).

Compare `REAL_MS_PER_ITER` values between `baseline` and `kiss_pick` variants.

## 10. Full raw data

Every candidate kiss proposed on every problem is in `raw_experiments/*/candidates.jsonl`.
- Each line = one candidate: `{"n": <#>, "code": "<python>", "result": {"ok": bool, "sim_time_us": float, "error": str, ...}}`
- Every rejected candidate shows WHY (mock env limitation, HW gate failure, correctness fail, etc.)
- Look for the successful pattern by finding the FIRST OK candidate with a low sim time

For xor_grid_bcast opus-4-8 (`raw_experiments/novel_kiss_xor_grid_bcast/candidates.jsonl`):
- n=1: `torch.bitwise_xor(ii, jj)` → CRASH (mock env doesn't have bitwise_xor)
- n=2: `ii ^ jj` → CRASH (^ operator not on TrackedTensor)  
- n=3: bit-by-bit reconstruction via `//` and `%` → PASS at 72us

Sonnet-5 found the algebraic form (`bit_i + bit_j - 2*bit_i*bit_j`) which sims 5us faster at 53us.

## 11. Known limitations

- **64-rank RT crashes remain** — some kiss/strat candidates that pass 2-rank HW gate crash at 64-rank with `signature check from peer` errors (Neuron compiler shape-inference gap across nodes). Not fixed in this session.
- **fable-5 regresses on some problems** — `fable-5` on `leftpad_bcast` (a trivial problem) defaults to trying collective optimizations instead of local closed-forms. Do not swap it in for kiss without further testing.
- **sonnet-4-6 requires Marketplace subscription** — our IAM role couldn't invoke it. If you have subscription access, it may be worth testing.
- **Popcount / bit manipulation problems** are hard for kiss to solve consistently — kiss needs 3-4 tries (each with a different sub-op that fails in mock env) before finding a working formulation. Give at least `--max-budget 15.0 --max-steps 50` for these.

## 12. Directory layout in this branch

- `FULL_SCOREBOARD.md` — full head-to-head results with problem descriptions and solution snippets
- `REPRODUCE.md` — this file
- `pipeline_code/` — patched framework files (kiss_phase3.py, kiss_token_shim.py, score_service_v2.py, hw_gate_run.py, prompt, novel problem definitions)
- `raw_experiments/` — every candidate kiss and every strat winner on every problem tested (candidates.jsonl, kiss_summary.json, run.log, results.json)
- `runtimes/kiss_xor_grid.py` — the runtime file for kiss's xor_grid win (the only confirmed kiss > strat under strict discipline)
- `rt_xor/` — real-training logs and summary for xor_grid_bcast (baseline vs kiss)
- `kiss_hwtest.log`, `novel_test.log`, `sonnet5_sweep.log`, `novel_v5_test.log` — master orchestration logs
