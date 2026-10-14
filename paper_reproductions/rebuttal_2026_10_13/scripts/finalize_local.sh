#!/bin/bash
# finalize_local.sh: pull every result dir from the master, recompute the final correlation table, refresh the artifact
# branch (scrubbed) and push. Run on the laptop.
set -uo pipefail
MSG=${1:-"PPoPP27 rebuttal: update artifacts"}
cd /tmp/ap7 && source cluster.env
SSHO="-i /Users/kekaiyao/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
bash pull_results.sh >/dev/null 2>&1
for d in sweep_a2av sweep_a2av_r2 sweep_a2av_r3 sweep_a2av_r4 sweep_found sweep_found_r2 sweep_found_r3 sweep_found2_r1 sweep_ua2a_fix sweep_ua2a_fix_r2 sweep_ua2a_fix_r3 gs_v4 hwprobe livecal sweep_livecal_r1 sweep_livecal_r2 sweep_livecal_r3; do
  mkdir -p pull/$d; rsync -az -e "ssh $SSHO" ubuntu@$MASTER_PUB:/home/ubuntu/$d/ pull/$d/ 2>/dev/null; done
scp -q $SSHO ubuntu@$MASTER_PUB:/home/ubuntu/queue.log pull/queue.log 2>/dev/null
EX=/tmp/ap7/pull/sweep_a2av,/tmp/ap7/pull/sweep_a2av_r2,/tmp/ap7/pull/sweep_a2av_r3,/tmp/ap7/pull/sweep_a2av_r4,/tmp/ap7/pull/sweep_found,/tmp/ap7/pull/sweep_found_r2,/tmp/ap7/pull/sweep_found_r3,/tmp/ap7/pull/sweep_ua2a_fix,/tmp/ap7/pull/sweep_ua2a_fix_r2,/tmp/ap7/pull/sweep_ua2a_fix_r3
EXTRA_DIRS=$EX SIM=/tmp/ap7/pull/sweep_v2/sim_v4_v13.json python3 analyze_v2.py > pull/analysis_final_v13.txt 2>&1
sed -n '4,16p' pull/analysis_final_v13.txt
# refresh artifact branch
cd /tmp/acc_push && git checkout -q ppopp27-rebuttal-2026-10-13 && D=paper_reproductions/rebuttal_2026_10_13 && rm -rf $D/scripts $D/results $D/docs $D/sim && mkdir -p $D/scripts $D/results $D/docs $D/sim
cd /tmp/ap7 && cp *.sh *.py manifest*.json /tmp/acc_push/$D/scripts/ 2>/dev/null; rm -f /tmp/acc_push/$D/scripts/correctness_test_v4.py /tmp/acc_push/$D/scripts/cluster.env
cp -r cands cands_v2 cands_found cands_found2 cands_a2av /tmp/acc_push/$D/scripts/ 2>/dev/null || true
cp correctness_test_v4.py /tmp/acc_push/$D/sim/; cp *.md /tmp/acc_push/$D/docs/; cp -r pull/* /tmp/acc_push/$D/results/
cd /tmp/acc_push/$D && find . -type f ! -name '*.tgz' -print0 | xargs -0 perl -pi -e 's/\b54\.79\.\d+\.\d+\b/<MASTER-EIP>/g; s/ip-172-31-\d+-\d+/ip-<PRIV>/g; s/\b172\.31\.\d+\.\d+\b/<PRIV-IP>/g; s/ghp_[A-Za-z0-9]+/<REDACTED>/g'
echo "scrub check: $(grep -rlE '54\.79\.|172\.31\.|ghp_' . | wc -l) files still match"
cd /tmp/acc_push && git add -A $D && GIT_AUTHOR_NAME="Kaiyao Ke" GIT_AUTHOR_EMAIL="kaiyaoke@berkeley.edu" GIT_COMMITTER_NAME="kaiyaok2" GIT_COMMITTER_EMAIL="kaiyaoke@berkeley.edu" git commit -q -m "${MSG}" && git log --oneline -1 | cut -c1-90
B64=$(printf 'x-access-token:%s' "$GH_PAT" | base64); git -c http.extraHeader="Authorization: Basic $B64" push -q origin ppopp27-rebuttal-2026-10-13 2>&1 | sed -E 's/ghp_[A-Za-z0-9]+/ghp_***/g; s/Basic [A-Za-z0-9+\/=]+/Basic ***/g' | tail -2; echo "push rc=$?"
