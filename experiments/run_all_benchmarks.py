#!/usr/bin/env python3
"""
Run all 4 problems through Real Hardware Benchmark (Phase 4/5),
optionally with KISS Sorcar ablation, and optionally follow up
with 2-node convergence training using agent-generated algorithms.

Usage:
    python experiments/run_all_benchmarks.py [--kiss-sorcar] [--all]
    python experiments/run_all_benchmarks.py --all --train
    python experiments/run_all_benchmarks.py --train-only --train-steps 200
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT / "experiments" / "results"
PYTHON = sys.executable

PROBLEMS = ["alltoallv", "uniform_a2a", "fused_reducescatter", "ring_kv"]
MASTER_ADDR = "172.31.48.122"
WORKER_ADDRS = "172.31.55.245"
NEURON_VENV = "/opt/aws_neuronx_venv_pytorch_2_9"

TRAINING_SCRIPTS = {
    "alltoallv": "training/train.py",
    "uniform_a2a": "training/train_uniform_a2a.py",
    "fused_reducescatter": "training/train_fused_reducescatter.py",
    "ring_kv": "training/train_ring_kv.py",
}

TRAINING_BASELINES = {
    "alltoallv": "agrs",
    "uniform_a2a": "baseline",
    "fused_reducescatter": "baseline",
    "ring_kv": "baseline",
}


def run_search(problem, kiss_sorcar=False, hw_eval=True, num_nodes=2):
    """Run the 5-phase pipeline for a single problem."""
    cmd = [
        PYTHON, str(PROJECT / "experiments" / "run_search.py"),
        "--problem", problem,
        "--pattern", "moe",
        "--llm-model", "opus",
        "--output-dir", str(RESULTS_DIR),
        "--num-nodes", str(num_nodes),
        "--master-addr", MASTER_ADDR,
        "--worker-addrs", WORKER_ADDRS,
    ]
    if hw_eval:
        cmd.append("--hw-eval")
    if kiss_sorcar:
        cmd.append("--kiss-sorcar")

    tag = "sorcar" if kiss_sorcar else "normal"
    print(f"\n{'='*70}")
    print(f"  Running: {problem} ({tag})")
    print(f"{'='*70}")
    print(f"  Command: {' '.join(cmd)}")

    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=3600,
        cwd=str(PROJECT))
    dt = time.time() - t0

    log_dir = RESULTS_DIR / "benchmark_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{problem}_{tag}.log"
    log_path.write_text(
        f"=== STDOUT ===\n{result.stdout}\n"
        f"=== STDERR ===\n{result.stderr}\n"
        f"=== EXIT: {result.returncode} | TIME: {dt:.0f}s ===\n")

    print(f"  Exit: {result.returncode} | Time: {dt:.0f}s")
    if result.returncode != 0:
        print(f"  STDERR (last 500): {result.stderr[-500:]}")
    else:
        for line in result.stdout.split('\n'):
            if 'winner' in line.lower() or 'best' in line.lower() or 'hw:' in line.lower():
                print(f"  {line.strip()}")

    return {
        "problem": problem,
        "mode": tag,
        "phase": "search",
        "exit_code": result.returncode,
        "time_s": round(dt, 1),
        "stdout_tail": result.stdout[-1000:] if result.stdout else "",
    }


def run_training(problem, backend, num_nodes=2, steps=5000, warmup=5):
    """Run 2-node convergence training for a single problem + backend."""
    script = TRAINING_SCRIPTS.get(problem)
    if not script:
        print(f"  No training script for {problem}")
        return None

    port = 29600 + list(TRAINING_SCRIPTS.keys()).index(problem) * 10 + \
        {"agent": 0, "evolved": 1, "agrs": 2, "baseline": 3, "sorcar": 4}.get(backend, 5)

    torchrun_args = (
        f"--nnodes={num_nodes} "
        f"--nproc_per_node=32 "
        f"--rdzv_backend=c10d "
        f"--rdzv_endpoint={MASTER_ADDR}:{port}"
    )
    train_args = f"--backend {backend} --steps {steps} --warmup {warmup}"

    print(f"\n  Training: {problem} backend={backend} steps={steps} port={port}")

    # Sync code to worker
    try:
        subprocess.run(
            ["bash", str(PROJECT / "experiments" / "sync_nodes.sh"), WORKER_ADDRS],
            capture_output=True, timeout=60, cwd=str(PROJECT))
    except Exception as e:
        print(f"  Sync warning: {e}")

    env_setup = (
        f"export PATH={NEURON_VENV}/bin:$PATH; "
        f"export NEURON_RT_NUM_CORES=32; "
        f"export NEURON_NUM_RECENT_MODELS_TO_KEEP=1; "
        f"export FI_PROVIDER=efa; "
        f"export FI_EFA_USE_DEVICE_RDMA=1; "
        f"export XLA_TRANSFER_SEED_ASYNC=1; "
        f"export NEURON_CC_FLAGS='--retry_failed_compilation'"
    )

    torchrun_cmd = f"{NEURON_VENV}/bin/torchrun {torchrun_args} {script} {train_args}"

    # Launch worker in background via SSH
    worker_cmd = (
        f"ssh -o StrictHostKeyChecking=no ubuntu@{WORKER_ADDRS} "
        f"'cd {PROJECT} && {env_setup} && "
        f"export MASTER_ADDR={MASTER_ADDR} && "
        f"export MASTER_PORT={port} && "
        f"{torchrun_cmd}'"
    )

    t0 = time.time()
    worker_proc = None
    try:
        worker_proc = subprocess.Popen(
            worker_cmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Launch master
        master_env = os.environ.copy()
        master_env["PATH"] = f"{NEURON_VENV}/bin:" + master_env.get("PATH", "")
        master_env["NEURON_RT_NUM_CORES"] = "32"
        master_env["NEURON_NUM_RECENT_MODELS_TO_KEEP"] = "1"
        master_env["FI_PROVIDER"] = "efa"
        master_env["FI_EFA_USE_DEVICE_RDMA"] = "1"
        master_env["XLA_TRANSFER_SEED_ASYNC"] = "1"
        master_env["NEURON_CC_FLAGS"] = "--retry_failed_compilation"
        master_env["MASTER_ADDR"] = MASTER_ADDR
        master_env["MASTER_PORT"] = str(port)

        result = subprocess.run(
            torchrun_cmd.split(),
            capture_output=True, text=True, timeout=7200,
            cwd=str(PROJECT), env=master_env)

        dt = time.time() - t0

        # Wait for worker
        try:
            worker_proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            worker_proc.kill()

        # Log
        log_dir = RESULTS_DIR / "training_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{problem}_{backend}_train.log"
        log_path.write_text(
            f"=== STDOUT ===\n{result.stdout}\n"
            f"=== STDERR ===\n{result.stderr}\n"
            f"=== EXIT: {result.returncode} | TIME: {dt:.0f}s ===\n")

        print(f"  Exit: {result.returncode} | Time: {dt:.0f}s")
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if any(k in line.lower() for k in ['avg', 'throughput', 'final', 'wall']):
                    print(f"  {line.strip()}")

        return {
            "problem": problem,
            "backend": backend,
            "phase": "training",
            "exit_code": result.returncode,
            "time_s": round(dt, 1),
            "stdout_tail": result.stdout[-1000:] if result.stdout else "",
        }

    except subprocess.TimeoutExpired:
        dt = time.time() - t0
        print(f"  TIMEOUT: {problem}/{backend} ({dt:.0f}s)")
        if worker_proc:
            worker_proc.kill()
        return {
            "problem": problem, "backend": backend, "phase": "training",
            "exit_code": -1, "time_s": round(dt, 1), "error": "timeout",
        }
    except Exception as e:
        dt = time.time() - t0
        print(f"  ERROR: {e}")
        if worker_proc:
            worker_proc.kill()
        return {
            "problem": problem, "backend": backend, "phase": "training",
            "exit_code": -1, "time_s": round(dt, 1), "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kiss-sorcar", action="store_true",
                        help="Run with Sorcar ablation only")
    parser.add_argument("--normal", action="store_true",
                        help="Run normal pipeline only")
    parser.add_argument("--all", action="store_true",
                        help="Run both normal and Sorcar")
    parser.add_argument("--problems", nargs="+", default=PROBLEMS,
                        choices=PROBLEMS)
    parser.add_argument("--no-hw-eval", action="store_true",
                        help="Skip hardware evaluation (simulator only)")
    parser.add_argument("--num-nodes", type=int, default=2)
    parser.add_argument("--train", action="store_true",
                        help="Run 2-node training after search (agent vs baseline)")
    parser.add_argument("--train-only", action="store_true",
                        help="Skip search, only run training")
    parser.add_argument("--train-steps", type=int, default=5000,
                        help="Training steps per backend")
    args = parser.parse_args()

    modes = []
    if args.all:
        modes = [False, True]
    elif args.kiss_sorcar:
        modes = [True]
    else:
        modes = [False]

    all_results = []

    # Phase 1-5: Search + HW benchmark
    if not args.train_only:
        for kiss_sorcar in modes:
            for problem in args.problems:
                try:
                    r = run_search(
                        problem, kiss_sorcar=kiss_sorcar,
                        hw_eval=not args.no_hw_eval,
                        num_nodes=args.num_nodes)
                    all_results.append(r)
                except subprocess.TimeoutExpired:
                    print(f"  TIMEOUT: {problem}")
                    all_results.append({
                        "problem": problem,
                        "mode": "sorcar" if kiss_sorcar else "normal",
                        "phase": "search",
                        "exit_code": -1,
                        "time_s": 3600,
                        "error": "timeout",
                    })
                except Exception as e:
                    print(f"  ERROR: {e}")
                    all_results.append({
                        "problem": problem,
                        "mode": "sorcar" if kiss_sorcar else "normal",
                        "phase": "search",
                        "exit_code": -1,
                        "error": str(e),
                    })

    # Training: agent (from runtime/) vs baseline
    if args.train or args.train_only:
        print(f"\n{'='*70}")
        print(f"  Training Phase: Agent-Generated vs Baseline")
        print(f"  Steps: {args.train_steps} | Nodes: {args.num_nodes}")
        print(f"{'='*70}")
        for problem in args.problems:
            baseline = TRAINING_BASELINES[problem]
            for backend in ["agent", baseline]:
                try:
                    r = run_training(
                        problem, backend,
                        num_nodes=args.num_nodes,
                        steps=args.train_steps)
                    if r:
                        all_results.append(r)
                except Exception as e:
                    print(f"  Training ERROR: {problem}/{backend}: {e}")
                    all_results.append({
                        "problem": problem,
                        "backend": backend,
                        "phase": "training",
                        "exit_code": -1,
                        "error": str(e),
                    })

    summary_path = RESULTS_DIR / "benchmark_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n{'='*70}")
    print(f"  Summary saved to {summary_path}")
    print(f"{'='*70}")
    for r in all_results:
        phase = r.get("phase", "search")
        status = "OK" if r.get("exit_code") == 0 else "FAIL"
        label = r.get("mode", r.get("backend", "?"))
        print(f"  {r['problem']:25s} {phase:8s} {label:8s} {status:4s} {r.get('time_s', '?')}s")


if __name__ == "__main__":
    main()
