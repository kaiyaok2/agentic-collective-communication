# Offline (archive-only) sim-vs-measured evidence — available now, independent of the new cluster

Measured = 224-rank training scope (per-problem harness `avg_ms`, or OLMoE whole-step ms for AllToAllV).
Sim = the paper run's calibrated simulator (Phase-3 pool / search-log values). Only within-problem
comparisons are meaningful (different problems have different units/scopes).

| Problem | Candidate A (sim µs) | Candidate B (sim µs) | sim says | measured A | measured B | agree? |
|---|---|---|---|---|---|---|
| AllToAllV (OLMoE step) | ag_slice_cat (5432.6) | AG+T+RS (6794.7) | A | 3123–3140 ms (n=6) | 4392–4404 ms (n=7) | yes (1.25× sim vs 1.41× meas) |
| Ring KV | per_slot_allgather (5381) | per_chunk_per_part (11514) | A | 10.39–10.85 ms (n=3 styles) | 13.92–14.45 ms (n=4) | yes (2.14× vs 1.33×) |
| Layer-block AR | stacked single AR (5982.5) | two sequential AR (6046.5) | A | 27.7–28.3 ms (n=3) | 30.1–31.1 ms (n=4) | yes (1.01× vs 1.08×) |
| TP MLP | flatten single AR (18936) | per_mb (69868.8) | A | 23.0 ms (n=1, no-sim run) | 23.8 ms (n=1) | yes in sign (3.7× vs 1.03×) |
| dxe | 2-AR stacked (5525.2) | full_vocab_ag baseline (—) | A | Table 5: 14.63 ms/step contrib | 16.74 | yes |
| PP, FSDP, Uniform A2A | paper winner | baseline | A | Table 5 7n-train | Table 5 | yes (sign) |

Sign agreement 8/8 (the simulator's preferred strategy is the measured winner in every problem);
magnitude is not predictive (ratios differ 2–3× either way), which matches the paper's own framing
("ranking, not absolute time"). Cross-scope inversion evidence already in the paper: Table 5 1n-bench
column prefers the baseline for AllToAllV, Uniform A2A, PP, TP MLP, FSDP, Layer-block (6/8) while
training scope prefers the agent in 8/8 — the simulator agrees with training scope in 8/8.

Caveats for the response: (i) TP MLP per-problem-harness gain (1.03×) is far below Table 5's e2e
per-call contribution ratio (8.04→1.93) because the per-problem harness runs both backends inside one
`mark_step` pair; Table 5's Llama numbers come from the e2e probes. (ii) Uniform A2A agent crashed
(SIGABRT) at 224-rank training scope in r28 (all three styles), r33 and r42; the single success is the
r28_ua2a_debug run. Expect the same in the new sweep; report it under rejection/fragility.

Repeats under the paper protocol (C-Q5): OLMoE baseline n=7 → 4391.7, 4397.5, 4398.3, 4398.6, 4399.5,
4399.8, 4400.8, 4404.3 ms; agent n=6 → 3123.1, 3125.8, 3125.8, 3127.0, 3128.0, 3139.8 ms.
Gate outcomes (C8): 125 archived Phase-4b candidates → 93 pass / 15 fail / 17 skip.
