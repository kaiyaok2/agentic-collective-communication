#!/bin/bash
# Launch script for tomorrow's CB (cr-0d7ee22e9c58ec7b3).
# Assumes new master already reachable via ssh trn7-new alias.
# Bootstraps environment, applies patches, then runs full sweep autonomously.
set -eu

# --- 1. Provision master + worker ---
MASTER_IP=$(hostname -I | awk '{print $1}')
# Worker IP is passed as $1
WORKER_IP="${1:-}"
if [ -z "$WORKER_IP" ]; then
    echo "usage: launch_tomorrow.sh <WORKER_IP>"
    exit 1
fi
echo "master=$MASTER_IP worker=$WORKER_IP"

# --- 2. Clone repos ---
cd /home/ubuntu
rm -rf cb2_verify agentic-collective-communication
mkdir -p cb2_verify
git clone https://github.com/OverlayCCL/OverlayCCL.git cb2_verify/repo
# GH_PAT must be exported in the session before running this script.
git clone "https://${GH_PAT}@github.com/kaiyaok2/agentic-collective-communication.git"
cd agentic-collective-communication && git checkout v4-signature-fix-2026-07-24

# --- 3. Apply patches ---
bash /home/ubuntu/agentic-collective-communication/bootstrap_v6/apply.sh /home/ubuntu/cb2_verify/repo

# --- 4. Push SSH key to worker ---
[ -f ~/.ssh/id_rsa ] || ssh-keygen -q -t rsa -f ~/.ssh/id_rsa -N ""
PUBKEY=$(cat ~/.ssh/id_rsa.pub)
ssh -i ~/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    ubuntu@$WORKER_IP "echo \"$PUBKEY\" >> ~/.ssh/authorized_keys"

# --- 5. Copy Python scripts to worker ---
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    /home/ubuntu/hw_gate_run.py /home/ubuntu/rt_run_v12.py ubuntu@$WORKER_IP:/home/ubuntu/

# --- 6. Install kiss venv + boto3 ---
which python3.13 >/dev/null 2>&1 || {
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get install -y python3.13 python3.13-venv
}
mkdir -p /home/ubuntu/kiss
python3.13 -m venv /home/ubuntu/kiss/.venv
/home/ubuntu/kiss/.venv/bin/pip install --quiet boto3 anthropic

# --- 7. Verify all challenge baselines pass correctness ---
export PATH=/opt/aws_neuronx_venv_pytorch_2_9/bin:$PATH
export ACC_REPO=/home/ubuntu/cb2_verify/repo
export USE_BEDROCK=1 BEDROCK_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1

/opt/aws_neuronx_venv_pytorch_2_9/bin/python - << 'PYEOF' > /home/ubuntu/verify_baselines.log 2>&1
import sys, os
sys.path.insert(0, "/home/ubuntu/cb2_verify/repo")
os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")
import experiments.run_search as RS
import search.problems_comm_v7
import search.problems_challenge_v8
from search.problems import get_problem
from search.correctness_test import test_xla_candidate_generic
from search.template_evolution import TemplateEvolution
from search.contention_analysis import ContentionAnalyzer

agent_sim, topology, dispatch_overhead = RS.phase1_profiling(use_llm=False, llm_model="sonnet", num_nodes=2, verbose=False)
send_counts = RS.make_send_counts("moe", world=topology.num_cores)
ca = ContentionAnalyzer(topology, send_counts)

CHALLENGE = ["multi_grad_ar_chal", "ag_then_rs_chal", "multi_layer_ar_chal", "double_reduction_chal", "hierarchical_ar_chal", "sparse_topk_chal", "weighted_mean_chal", "layered_matmul_chal", "mixed_precision_ar_chal", "rotating_shuffle_chal", "batched_ar_scale_chal"]
for name in CHALLENGE:
    try:
        problem = get_problem(name)
        te = TemplateEvolution(topology, send_counts, agent_sim, ca, model="opus", problem=problem, unsupported_primitives=agent_sim.config.unsupported_primitives)
        baseline_code = list(problem.builtin_templates.values())[0]
        fn = te._sandbox_exec(baseline_code, is_nki=False)
        ok, det = test_xla_candidate_generic(problem, fn, num_nodes=2, unsupported_primitives=agent_sim.config.unsupported_primitives)
        print(f"{name:32}: baseline correctness={ok}")
    except Exception as e:
        print(f"{name:32}: FAIL {str(e)[:120]}")
PYEOF
echo "=== Baseline verification ==="
cat /home/ubuntu/verify_baselines.log | tail -20

# --- 8. Full strat sweep: 2 rep _bcast + 10 comm + 10 challenge + 8 OverlayCCL ---
mkdir -p /home/ubuntu/results/strat
cat > /tmp/sweep_strat_v13.sh << 'STRATEOF'
#!/bin/bash
set -u
cd /home/ubuntu/cb2_verify/repo
export PATH=/opt/aws_neuronx_venv_pytorch_2_9/bin:$PATH
export ACC_REPO=/home/ubuntu/cb2_verify/repo
export USE_BEDROCK=1 BEDROCK_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1

PROBLEMS=(
  xor_grid_bcast gray_code_bcast
  sum_across_ranks_comm max_across_ranks_comm concat_all_ranks_comm dot_across_ranks_comm shift_neighbor_comm reduce_scatter_sum_comm mean_max_normalize_comm rank_prefix_sum_comm center_by_mean_comm top_k_scalars_comm
  multi_grad_ar_chal ag_then_rs_chal multi_layer_ar_chal double_reduction_chal hierarchical_ar_chal sparse_topk_chal weighted_mean_chal layered_matmul_chal mixed_precision_ar_chal rotating_shuffle_chal batched_ar_scale_chal
  alltoallv uniform_a2a ring_kv grad_ar dxe pp_send_recv tp_mlp fsdp_prefetch llama_block_ar
)
for prob in "${PROBLEMS[@]}"; do
  outdir=/home/ubuntu/results/strat/$prob
  mkdir -p "$outdir"
  T0=$(date +%s)
  timeout 400 /opt/aws_neuronx_venv_pytorch_2_9/bin/python /home/ubuntu/cb2_verify/repo/experiments/run_search.py \
    --problem $prob --pattern moe --output-dir $outdir \
    --num-nodes 2 --phase3-style strategy-enumerate --llm-model opus \
    --max-rounds 2 --generations 1 --population 1 > $outdir/run.log 2>&1
  T1=$(date +%s); DT=$((T1-T0))
  BEST=$(grep 'Simulator winner' $outdir/run.log 2>/dev/null | grep -oE 'sim=[0-9.]+' | head -1)
  echo "[strat] $prob = $BEST (${DT}s)" | tee -a /home/ubuntu/results/strat_summary.txt
done
STRATEOF
chmod +x /tmp/sweep_strat_v13.sh
nohup bash /tmp/sweep_strat_v13.sh > /home/ubuntu/results/strat_master.log 2>&1 &
echo "strat sweep launched, PID=$!"

# --- 9. Also commit a status marker so we know progress ---
cd /home/ubuntu/agentic-collective-communication
echo "Tomorrow launch script executed at $(date -u)" > TOMORROW_LAUNCH.txt
git add TOMORROW_LAUNCH.txt
GIT_AUTHOR_NAME='Kaiyao Ke' GIT_AUTHOR_EMAIL='kaiyaoke@berkeley.edu' \
GIT_COMMITTER_NAME='kaiyaok2' GIT_COMMITTER_EMAIL='kaiyaoke@berkeley.edu' \
git commit -m 'Tomorrow launch marker' || true
git push "https://${GH_PAT}@github.com/kaiyaok2/agentic-collective-communication.git" v4-signature-fix-2026-07-24 || true

echo "=== Launch bootstrap complete. Strat sweep running in background. ==="
