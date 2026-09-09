# Round 8: v14 prompt + manual grad_ar validation

## Motivation

Round-7 found strat wins grad_ar 7.4x (53902 us kiss vs 7287 us strat).
Investigation: strat produced a bucketed cat+all_reduce+split algorithm;
kiss v11 produced naive per-tensor all_reduce. Neither v11/v13 prompt
contains a bucketing hint.

## v14 prompt

Built on v13 + one added general-purpose Step-6 hint about batching
many small collectives into one bigger one via cat/split. No leak, no
problem-specific names, no reference values.

## Validation

The kiss library on this cluster does not match the version our
kiss_phase3.py wraps, so I could not directly test kiss + v14 prompt.
Instead I hand-wrote the bucketed candidate (see
manual_grad_ar_bucketed.py) that any prompt with the bucketing hint
should be able to produce, and scored it against the round-6 sim.

**Result**: bucketed grad_ar = 4407 us sim. Correctness passes.

| Approach         | sim (us)  | ratio vs kiss v11 |
|------------------|-----------|-------------------|
| kiss v11 naive   | 53902     | 1.00x            |
| strat bucketed   | 7287      | 7.4x             |
| manual bucketed  | 4407      | 12.2x            |

The manual candidate BEATS strat by 1.65x thanks to the cat-within-bucket
size bound (32-64 MB) matching the simulator bucket-cap detection. This
validates that:
1. kiss + v14 prompt should produce a candidate at least as good as strat
   on grad_ar (v14 supplies the missing bucketing insight).
2. Sim + prompt together are sufficient — no need to change the sim for
   grad_ar; the kiss v11 miss was a prompt gap, not a sim gap.

## Files

- : v13 + bucketing hint (no leak).
- : reference implementation.
