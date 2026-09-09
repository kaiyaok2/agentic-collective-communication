#!/bin/bash
# Push E2E 10B results to the existing branch. PAT as $1 (never echoed).
set -u
PAT="${1:?usage: push_e2e.sh <PAT>}"
REPO=/home/ubuntu/acc
BRANCH="taxonomy-3col-rt-2026-09-09"
cd "$REPO" || exit 1
REMOTE="https://${PAT}@github.com/kaiyaok2/agentic-collective-communication.git"

# fetch the branch (it was first created from the temp restore instance)
git fetch "$REMOTE" "$BRANCH" 2>&1 | sed -E 's/ghp_[A-Za-z0-9]+/ghp_***/g'
git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" FETCH_HEAD 2>/dev/null
git reset --soft FETCH_HEAD 2>/dev/null

git add e2e_10b_results
GIT_AUTHOR_NAME="Kaiyao Ke" GIT_AUTHOR_EMAIL="kaiyaoke@berkeley.edu" \
GIT_COMMITTER_NAME="kaiyaok2" GIT_COMMITTER_EMAIL="kaiyaoke@berkeley.edu" \
git commit -q -m "E2E 10B-TP: Sorcar vs baseline, multi-seed x both-arch >=2.0x

7-node trn1, WS=224 (TP=32 x DP=7), 48 layers, N_MB=16 --fuse, warm cache.
Llama 2.49x (seed 42/43), GPT 2.20/2.22x (held-out arch); loss parity clean
(final deltas within per-step data-shuffle noise, F3 checksum==0). N_MB scaling
4->16 shows 1.67x->2.49x as per-microbatch resync compounds. Training scripts,
launcher, per-run JSON + master logs, and the research-loop ideas/explored docs."
echo "commit rc=$? sha=$(git rev-parse --short HEAD)"

git push "$REMOTE" "$BRANCH" 2>&1 | sed -E 's/ghp_[A-Za-z0-9]+/ghp_***/g'
echo "push rc=${PIPESTATUS[0]}"
