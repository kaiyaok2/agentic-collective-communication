# Warm-RT sweep runbook — Sorcar vs Overlay 9-seed medians (fair-divergence families)

CB `cr-0e709fd8e24812597`, us-east-1c, 7× trn1.32xlarge = 224 ranks (matches sim
world_size=224, num_devices=112, cores_per_device=2, num_nodes=7). Start 2026-09-24
01:08 UTC. Compares each staged problem's **median (middle) seed** Sorcar code vs
Overlay code under warm Neuron cache.

## Staged set
`warmrt/stage/<fam>/<prob>/{sorcar.py,overlay.py}` — 49 problems (fam1=3, fam2=7,
fam3=5, fam4=2, fam5=10, fam6=11, fam7=11), each the 9-seed lower-median by sim
(kiss `best_sim_time_us`, overlay `final_sim`). `stage/master_manifest.json` records
the chosen seeds + sim ratios.

## Fire sequence (once CB State=active)
1. **Laptop:** `AWS_PROFILE=kaiyao bash warmrt/launch_cb7.sh`  → writes
   `/tmp/warmrt_cluster.env` (MASTER_PUB, MASTER_PRIV, WORKER_PRIVS). Associates EIP.
2. **Laptop → master:** copy key + export PAT + clone, then bootstrap:
   ```
   source /tmp/warmrt_cluster.env
   scp -i ~/.ssh/Kaiyao.pem ~/.ssh/Kaiyao.pem ubuntu@$MASTER_PUB:/home/ubuntu/.ssh/Kaiyao.pem
   ssh -i ~/.ssh/Kaiyao.pem ubuntu@$MASTER_PUB \
     "chmod 600 ~/.ssh/Kaiyao.pem; export GH_PAT=<PAT>; \
      git -c http.extraHeader=\"Authorization: Basic \$(printf 'x-access-token:%s' \$GH_PAT|base64|tr -d '\n')\" \
        clone https://github.com/kaiyaok2/agentic-collective-communication.git /home/ubuntu/agentic-collective-communication; \
      export GH_PAT=<PAT>; bash /home/ubuntu/agentic-collective-communication/warmrt/bootstrap_master.sh \"\$WORKER_PRIVS\""
   ```
   (bootstrap_master.sh clones/updates the repo on all 6 workers + wires master→worker SSH.)
3. **Master:** run the sweep (median of 3 warm reps/problem/pipeline @ 50 iters):
   ```
   source /tmp/warmrt_cluster.env  # or set MASTER_PRIV / WORKER_PRIVS
   nohup bash /home/ubuntu/agentic-collective-communication/warmrt/run_warmrt_sweep.sh \
     "$MASTER_PRIV" "$WORKER_PRIVS" 50 3 > /home/ubuntu/warmrt_sweep.log 2>&1 &
   ```
   Results stream to `/home/ubuntu/warmrt_results/results.jsonl` + a speedup table at the end.
4. **Collect:** scp `warmrt_results/results.jsonl` + `warmrt_sweep.log` back; commit under
   `warmrt/results/`. Then **terminate** the 7 instances (CB keeps billing regardless, but
   free them when done): `aws ec2 terminate-instances --region us-east-1 --instance-ids $MASTER_ID $WORKER_IDS`.

## Notes
- `rt_diverge.py` loads the problem from the repo registry (glob-imports all
  `search/problems_diverge_*`), derives per-rank shard `part` from a world-2 test case,
  builds rank-r input `randn(world*part)*(0.3+0.02*r)`, and times the candidate inside a
  2-layer MLP so XLA can't DCE the collective. Emits `RT_TIME_MS_PER_ITER`.
- Warm-cache discipline: first rep warms the Neuron compile cache; median taken over
  reps 2..N (see rt_warm_cache_pitfall). If a run FAILs, rt_7node.sh dumps the rank-0 tail.
- If `run-instances --count 1` × 7 hits multi-NIC quota, launch fewer NICs or stagger.
