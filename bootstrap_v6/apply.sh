#!/bin/bash
# Bootstrap script: applies v6 patches to a fresh OverlayCCL clone.
# Usage: bash apply.sh /path/to/OverlayCCL/clone
set -eu
TARGET="${1:-/home/ubuntu/cb2_verify/repo}"
BOOTSTRAP="$(dirname "$(readlink -f "$0")")"

echo "Applying v6 patches to $TARGET"

# Overwrite the modified sim + phase-3 files
cp $BOOTSTRAP/search/correctness_test.py $TARGET/search/correctness_test.py
cp $BOOTSTRAP/search/agent_simulator_config.py $TARGET/search/agent_simulator_config.py
cp $BOOTSTRAP/experiments/run_search.py $TARGET/experiments/run_search.py

# Add new problem catalogs (post-paper additions)
cp $BOOTSTRAP/search/problems_novel_v4.py $TARGET/search/problems_novel_v4.py
cp $BOOTSTRAP/search/problems_novel_v5.py $TARGET/search/problems_novel_v5.py
cp $BOOTSTRAP/search/problems_novel_v6.py $TARGET/search/problems_novel_v6.py
cp $BOOTSTRAP/search/problems_kiss_verify.py $TARGET/search/problems_kiss_verify.py
cp $BOOTSTRAP/search/problems_comm_v7.py $TARGET/search/problems_comm_v7.py
cp $BOOTSTRAP/search/problems_challenge_v8.py $TARGET/search/problems_challenge_v8.py
cp $BOOTSTRAP/search/problems_round17.py $TARGET/search/problems_round17.py

# Add kiss integration
mkdir -p $TARGET/experiments/ablation_kiss_vs_cc
cp $BOOTSTRAP/experiments/ablation_kiss_vs_cc/score_service_v2.py $TARGET/experiments/ablation_kiss_vs_cc/
cp $BOOTSTRAP/experiments/ablation_kiss_vs_cc/kiss_phase3.py $TARGET/experiments/ablation_kiss_vs_cc/

# Prompts
cp $BOOTSTRAP/prompts/generic_evolution.md $TARGET/prompts/generic_evolution.md
cp $BOOTSTRAP/prompts/generic_evolution_v11.md $TARGET/prompts/generic_evolution_v11.md
cp $BOOTSTRAP/prompts/generic_evolution_v13.md $TARGET/prompts/generic_evolution_v13.md
cp $BOOTSTRAP/prompts/generic_evolution_v14.md $TARGET/prompts/generic_evolution_v14.md

# RT / HW gate scripts (place in ubuntu home for direct invocation)
cp $BOOTSTRAP/rt_run_v12.py /home/ubuntu/rt_run_v12.py
cp $BOOTSTRAP/rt_2node.sh /home/ubuntu/rt_2node.sh
cp $BOOTSTRAP/hw_gate_run.py /home/ubuntu/hw_gate_run.py
cp $BOOTSTRAP/hw_gate_2node.sh /home/ubuntu/hw_gate_2node.sh

# Verify syntax
python3 -c "import ast; ast.parse(open('$TARGET/search/correctness_test.py').read())"
python3 -c "import ast; ast.parse(open('$TARGET/search/agent_simulator_config.py').read())"
python3 -c "import ast; ast.parse(open('$TARGET/experiments/run_search.py').read())"

echo 'Patches applied. Target: '$TARGET
