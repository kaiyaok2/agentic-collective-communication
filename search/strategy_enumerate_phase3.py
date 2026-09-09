"""Strategy-Enumeration + Targeted-Refinement Phase 3.

Innovation: rather than a single ReAct trajectory or K parallel
island-GA trajectories, the LLM is FORCED to first articulate K
distinct STRUCTURAL strategies (e.g., 'one stacked AR over packed
payload', 'M sequential ARs per microbatch', 'AG+RS chain with
host-side index_select'). It then implements each strategy as a
candidate, the simulator ranks them, and the top-2 are refined
through a few targeted-mutation rounds where the LLM is told which
cost term currently dominates.

Same LLM (configurable, defaults to the multi-island choice), same
prompt template root, same simulator scoring, same correctness gate,
same Phase 4 HW gate and Phase 5 selection downstream. The only
change is the search shape inside Phase 3.

Total LLM-call budget: 1 (enumeration) + K (per-strategy implementation)
+ R*2 (top-2 refinement rounds). Default K=5, R=3 -> ~12 calls per
problem, matching the multi-island 4-islands x 2-rounds = 8 calls
within a 50% headroom for the enumeration overhead.
"""
from __future__ import annotations
import re
from typing import Dict, Any, List, Tuple

from .template_evolution import (
    TemplateEvolution,
    _check_xla_safety,
)
from .generate_algo import _invoke_bedrock


ENUMERATION_PROMPT = """\
You are designing collective communication algorithms for AWS Trainium
at world size {world_size}. The task is to enumerate distinct
STRUCTURAL strategies for the following problem before implementing
any code.

## Problem: {display_name}
Function signature:
```
{signature}
```
{signature_doc}

## Seed library (correct reference implementations, varying speed)
{reference_implementations}

## Hardware constraints + cost model
{optimization_hints}

## Your task — strategy enumeration

List exactly {k} DISTINCT structural strategies for solving this
problem. By "distinct" I mean: each strategy uses a meaningfully
different combination of primitives and/or local-op shape (different
collective count, different payload layout, different fusion-eligible
ordering). Two strategies that differ only in low-level details
(loop unrolling, variable naming, op renaming) count as one.

Output format (machine-parseable):
```
STRATEGY 1: <name, 3-6 words>
DESCRIPTION: <1-2 sentences on what primitives it uses, expected
dispatch count, and the cost-term tradeoff it represents (e.g.,
'fewer dispatches but larger per-dispatch payload', 'pipeline-fill
across N back-to-back ARs', 'AG+RS chain that hides O(N) memcpy in
metadata view ops')>

STRATEGY 2: ...
...

STRATEGY {k}: ...
```

Do NOT write any Python code yet. Just enumerate.
"""

IMPLEMENT_PROMPT = """\
You previously enumerated this strategy for the {display_name} problem:

STRATEGY: {strategy_name}
DESCRIPTION: {strategy_description}

## Problem signature
```
{signature}
```
{signature_doc}

## Cost model + hardware constraints
{optimization_hints}

## Reference implementations (correctness oracles, ALL are correct)
{reference_implementations}

## Your task

Write a Python implementation of the strategy above. Constraints:
- Implement the exact function signature {evolved_fn_name}.
- Use ONLY xm.collective_permute / xm.all_gather / xm.all_reduce /
  xm.reduce_scatter for communication.
- Use Python ints / lists for all index arithmetic (no device
  tensors as Python-level values).
- Preserve input dtype (read input_tensor.dtype, do not hardcode).

Wrap your code in a single ```python ... ``` block.
"""

REFINE_PROMPT_NO_SIM = """\
You are refining a candidate implementation of the {display_name}
problem. The simulator is not available in this ablation; you must
reason from code structure alone. After all refinement rounds the
deployed code will be picked by a small-shape hardware microbench
(per-call latency on the real cluster), so simpler patterns with
fewer Python-level loops and fewer collective dispatches are
favored at that stage. Optionally, write a tiny inline profiling
print (the cluster runs see stdout) inside the candidate so you
can self-measure between rounds; microbenchmarks alone can be
misleading at this small shape.

## Current candidate
```python
{current_code}
```

## Static op counts (no simulator, just AST counts)
- num_collective_permute = {num_collective_permute}
- num_all_gather = {num_all_gather}
- num_all_reduce = {num_all_reduce}
- local_ops = {local_ops}

## History of prior refinements (op-count deltas only)
{history}

## Your task
Mutate the candidate to reduce collective dispatch count and
local op count, favoring well-known compact patterns like AG+RS,
AG+T+RS, pack+AG+slice, etc. Keep the same signature. Wrap your
new code in a single ```python ... ``` block.
"""


REFINE_PROMPT_HEADER = """\
You are refining a candidate implementation of the {display_name}
problem. The simulator scored this candidate; the breakdown below
identifies the dominant cost term and suggests where to optimize.

## Current candidate
```python
{current_code}
```

## Simulator score: {sim_us:.1f} us
- num_collective_permute = {num_collective_permute}
- num_all_gather = {num_all_gather}
- num_all_reduce = {num_all_reduce}
- local_ops = {local_ops}
- dominant_term: {dominant_term}

{breakdown_text}

## History of prior refinements
{history}

## Your task
Mutate the candidate to reduce the dominant cost term {dominant_term}.
Keep the same signature. Wrap your new code in a single ```python ... ```
block.
"""


def _parse_strategies(response: str, k: int) -> List[Tuple[str, str]]:
    """Extract (name, description) pairs from the enumeration response."""
    out = []
    pattern = re.compile(
        r"STRATEGY\s*\d+\s*:\s*([^\n]+)\s*\n\s*DESCRIPTION\s*:\s*([^\n]+(?:\n(?!STRATEGY)[^\n]+)*)",
        re.IGNORECASE,
    )
    for m in pattern.finditer(response):
        out.append((m.group(1).strip(), m.group(2).strip()))
        if len(out) >= k:
            break
    return out


def _extract_code(response: str) -> str:
    """Extract the first ```python ... ``` block (or first ``` block)."""
    m = re.search(r"```(?:python)?\s*(.*?)```", response, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def _identify_dominant_term(bench: Dict[str, Any]) -> Tuple[str, str]:
    """Return (term_name, helpful_text) describing the dominant cost term."""
    # The simulator returns sim_time_us but not per-term breakdown directly.
    # Approximate it from the recorded counts and the local-cost field.
    sim_us = bench.get("sim_time_us", 0.0)
    coll_count = (bench.get("num_collective_permute", 0)
                  + bench.get("num_all_gather", 0)
                  + bench.get("num_all_reduce", 0))
    local_ops = bench.get("local_ops", 0)
    local_cost = bench.get("local_cost_us", 0.0)
    # Each collective amortized ~50 us in deep-pipeline regime;
    # approximate collective cost contribution.
    approx_coll_us = coll_count * 100.0  # conservative: full per-dispatch
    if coll_count >= 4:
        # back-to-back amortization: first 100us, rest amortized
        approx_coll_us = 100.0 + (coll_count - 1) * 20.0
    if approx_coll_us > local_cost * 1.5:
        return ("collective dispatch overhead",
                f"  Approx collective contribution: ~{approx_coll_us:.0f} us\n"
                f"  Local op contribution: ~{local_cost:.0f} us\n"
                f"  To reduce: collapse multiple collectives into one "
                f"(e.g., stack payloads + one AR; pack-and-AG instead of "
                f"M smaller ones).")
    elif local_cost > approx_coll_us * 1.5:
        return ("local op cost",
                f"  Local op contribution: ~{local_cost:.0f} us\n"
                f"  Approx collective contribution: ~{approx_coll_us:.0f} us\n"
                f"  To reduce: replace volume-scaled ops (index_select, "
                f"tensor-from-list) with metadata-only views (narrow, "
                f"slice, reshape on dense-contiguous storage). Avoid "
                f"stack/cat on large payloads.")
    else:
        return ("balanced (no single dominant term)",
                f"  Approx collective contribution: ~{approx_coll_us:.0f} us\n"
                f"  Local op contribution: ~{local_cost:.0f} us\n"
                f"  Try restructuring: different primitive ordering, "
                f"or a different layout that lets fuseable compute sit "
                f"adjacent to collectives.")


def _phase3_strategy_enumerate(problem, topology, send_counts, cost_model,
                                baseline_results, use_llm, llm_model,
                                num_nodes, max_rounds,
                                verbose=True, no_simulator=False, **kwargs):
    """Strategy-Enumeration + Targeted-Refinement Phase 3.

    no_simulator=True ablates the simulator: skip refinement (no per-op
    cost signal to refine on), and pick the final winner deterministically
    by enumeration order (first-correct-and-HW-valid) rather than by
    simulator score.
    """
    print("\n" + "=" * 70)
    print(f"[Phase 3 / Strategy-Enumerate] {problem.display_name}"
          + (" [NO-SIM ABLATION]" if no_simulator else ""))
    print("=" * 70)

    all_results = list(baseline_results)
    if not use_llm:
        print("  Skipping (--no-llm)")
        return all_results

    # K = number of strategies to enumerate. Default 5.
    K = 5
    # R = refinement rounds per top candidate. Total budget = 1 + K + 2*R.
    R = max(1, max_rounds // 2)
    print(f"  K={K} strategies, R={R} refinement rounds per top-2 (budget ~{1 + K + 2*R} LLM calls)")

    # Need a TemplateEvolution scaffold to reuse _sandbox_exec / bench / etc.
    from .template_evolution import _check_xla_safety  # local import safe
    from .correctness_test import (
        test_xla_candidate_generic, test_xla_candidate_bf16,
        benchmark_xla_candidate_generic,
    )
    from .island_evolution import ContentionAnalyzer

    analyzer = ContentionAnalyzer(topology, send_counts)
    te = TemplateEvolution(
        topology, send_counts, cost_model, analyzer,
        model=llm_model, problem=problem, **kwargs)

    # Build the prompts' template-context block once.
    ref_impls_block = "\n\n".join(
        f"### {name}:\n```python\n{code}\n```"
        for name, code in problem.builtin_templates.items()
    )
    opt_hints = problem.optimization_hints.replace(
        "{op_cost_table}", te._format_op_cost_table()
    ).replace(
        "{dispatch_overhead_us}", f"{te.dispatch_overhead_us:.0f}"
    )

    # --- Stage 1: enumerate strategies (1 LLM call) ---
    enum_prompt = ENUMERATION_PROMPT.format(
        world_size=te.world,
        display_name=problem.display_name,
        signature=problem.signature,
        signature_doc=problem.signature_doc,
        reference_implementations=ref_impls_block,
        optimization_hints=opt_hints,
        k=K,
    )
    try:
        enum_response = _invoke_bedrock(
            enum_prompt, model=te.model, temperature=0.8, max_tokens=4000)
    except Exception as e:
        print(f"  Enumeration LLM error: {e}; falling back to multi-island.")
        from experiments.run_search import phase3_generic_evolution
        return phase3_generic_evolution(
            problem, topology, send_counts, cost_model, baseline_results,
            use_llm, llm_model, num_nodes, max_rounds,
            verbose=verbose, **kwargs)

    strategies = _parse_strategies(enum_response, K)
    print(f"  Enumerated {len(strategies)} strategies:")
    for i, (name, desc) in enumerate(strategies):
        print(f"    {i+1}. {name}: {desc[:80]}{'...' if len(desc) > 80 else ''}")

    if not strategies:
        print("  No strategies parsed; falling back to multi-island.")
        from experiments.run_search import phase3_generic_evolution
        return phase3_generic_evolution(
            problem, topology, send_counts, cost_model, baseline_results,
            use_llm, llm_model, num_nodes, max_rounds,
            verbose=verbose, **kwargs)

    # --- Stage 2: implement each strategy (K LLM calls) ---
    candidates = []
    for i, (name, desc) in enumerate(strategies):
        impl_prompt = IMPLEMENT_PROMPT.format(
            display_name=problem.display_name,
            strategy_name=name,
            strategy_description=desc,
            signature=problem.signature,
            signature_doc=problem.signature_doc,
            optimization_hints=opt_hints,
            reference_implementations=ref_impls_block,
            evolved_fn_name=problem.evolved_fn_name,
        )
        try:
            r = _invoke_bedrock(impl_prompt, model=te.model,
                                 temperature=0.7, max_tokens=6000)
            code = _extract_code(r)
        except Exception as e:
            print(f"  Strategy {i+1} ({name}) LLM error: {e}")
            continue
        if code is None:
            continue
        warnings = _check_xla_safety(code)
        if warnings:
            print(f"  Strategy {i+1} XLA-unsafe: {warnings[0]}")
            continue
        candidate_fn = te._sandbox_exec(code)
        if candidate_fn is None:
            print(f"  Strategy {i+1} sandbox failed")
            continue
        # R9: try/except around candidate correctness+bench so a
        # malformed LLM emission (return-type mismatch, dtype bug,
        # etc.) is caught per-strategy instead of killing Phase 3.
        try:
            passed, details = test_xla_candidate_generic(
                problem, candidate_fn, num_nodes=te.num_nodes,
                unsupported_primitives=te.unsupported_primitives)
            if not passed:
                print(f"  Strategy {i+1} ({name}) INCORRECT: {details[:100]}")
                continue
            bf16_ok, _ = test_xla_candidate_bf16(
                problem, candidate_fn, num_nodes=te.num_nodes,
                unsupported_primitives=te.unsupported_primitives)
            if not bf16_ok:
                continue
            bench = benchmark_xla_candidate_generic(
                problem, candidate_fn, te.topo, te.send_counts, te.world,
                num_nodes=te.num_nodes, **te._generic_bench_kwargs())
            if "error" in bench:
                continue
            sim_us = bench["sim_time_us"]
        except Exception as e:
            print(f"  Strategy {i+1} ({name}) test/bench exception: "
                  f"{type(e).__name__}: {str(e)[:120]}")
            continue
        print(f"  Strategy {i+1} ({name}): sim={sim_us:.1f} us")
        candidates.append({
            "name": name, "desc": desc, "code": code,
            "bench": bench, "sim_us": sim_us,
        })

    if not candidates:
        print("  No correct candidates from strategy enumeration; "
              "falling back to multi-island.")
        from experiments.run_search import phase3_generic_evolution
        return phase3_generic_evolution(
            problem, topology, send_counts, cost_model, baseline_results,
            use_llm, llm_model, num_nodes, max_rounds,
            verbose=verbose, **kwargs)

    if no_simulator:
        # Iterative no-sim refinement: keep enumeration order; refine
        # ALL enumerated candidates (no simulator-driven top-K). The
        # LLM will use code-structure reasoning + op counts (no sim
        # scores) to refine.
        print(f"\n  [NO-SIM ITERATIVE] Refining all {len(candidates)} "
              f"correct candidates (sim_us not shown to LLM in refinement)")
        candidates_top = candidates[:5]
    else:
        candidates.sort(key=lambda c: c["sim_us"])
        print(f"\n  Top after enumeration:")
        for c in candidates[:2]:
            print(f"    {c['name']}: {c['sim_us']:.1f} us")
        candidates_top = candidates[:2]

    # --- Stage 3: refine the top-2 (R rounds each) ---
    for top_idx, c in enumerate(candidates_top):
        history_text = f"  initial enumeration: {c['name']} -> {c['sim_us']:.1f} us"
        current_code = c["code"]
        current_bench = c["bench"]
        for round_idx in range(R):
            if no_simulator:
                refine_prompt = REFINE_PROMPT_NO_SIM.format(
                    display_name=problem.display_name,
                    current_code=current_code,
                    num_collective_permute=current_bench.get("num_collective_permute", 0),
                    num_all_gather=current_bench.get("num_all_gather", 0),
                    num_all_reduce=current_bench.get("num_all_reduce", 0),
                    local_ops=current_bench.get("local_ops", "?"),
                    history=history_text,
                )
            else:
                dom_term, breakdown_text = _identify_dominant_term(current_bench)
                refine_prompt = REFINE_PROMPT_HEADER.format(
                    display_name=problem.display_name,
                    current_code=current_code,
                    sim_us=current_bench["sim_time_us"],
                    num_collective_permute=current_bench.get("num_collective_permute", 0),
                    num_all_gather=current_bench.get("num_all_gather", 0),
                    num_all_reduce=current_bench.get("num_all_reduce", 0),
                    local_ops=current_bench.get("local_ops", "?"),
                    dominant_term=dom_term,
                    breakdown_text=breakdown_text,
                    history=history_text,
                )
            try:
                r = _invoke_bedrock(refine_prompt, model=te.model,
                                    temperature=0.7, max_tokens=6000)
                new_code = _extract_code(r)
            except Exception as e:
                print(f"    top {top_idx+1} round {round_idx+1} LLM error: {e}")
                continue
            if new_code is None:
                continue
            if _check_xla_safety(new_code):
                continue
            new_fn = te._sandbox_exec(new_code)
            if new_fn is None:
                continue
            try:
                ok, _ = test_xla_candidate_generic(
                    problem, new_fn, num_nodes=te.num_nodes,
                    unsupported_primitives=te.unsupported_primitives)
                if not ok:
                    history_text += f"\n  refine round {round_idx+1}: INCORRECT"
                    continue
                bf16_ok, _ = test_xla_candidate_bf16(
                    problem, new_fn, num_nodes=te.num_nodes,
                    unsupported_primitives=te.unsupported_primitives)
                if not bf16_ok:
                    continue
                new_bench = benchmark_xla_candidate_generic(
                    problem, new_fn, te.topo, te.send_counts, te.world,
                    num_nodes=te.num_nodes, **te._generic_bench_kwargs())
                if "error" in new_bench:
                    continue
                new_us = new_bench["sim_time_us"]
            except Exception as e:
                print(f"    top {top_idx+1} round {round_idx+1} "
                      f"test/bench exception: {type(e).__name__}: "
                      f"{str(e)[:120]}")
                continue
            if no_simulator:
                ops_delta = (
                    (current_bench.get("num_all_gather", 0) +
                     current_bench.get("num_all_reduce", 0) +
                     current_bench.get("num_collective_permute", 0))
                    -
                    (new_bench.get("num_all_gather", 0) +
                     new_bench.get("num_all_reduce", 0) +
                     new_bench.get("num_collective_permute", 0))
                )
                history_text += (
                    f"\n  refine round {round_idx+1}: "
                    f"collectives total={new_bench.get('num_all_gather',0)+new_bench.get('num_all_reduce',0)+new_bench.get('num_collective_permute',0)}, "
                    f"local_ops={new_bench.get('local_ops','?')}")
                # In no-sim mode, accept if collective count strictly drops
                # or if local op count drops by >=10 (simpler structure).
                if ops_delta > 0 or (
                    isinstance(current_bench.get("local_ops"), int)
                    and isinstance(new_bench.get("local_ops"), int)
                    and current_bench["local_ops"] - new_bench["local_ops"] >= 10):
                    print(f"    top {top_idx+1} round {round_idx+1}: "
                          f"SIMPLER (collective Δ={ops_delta})")
                    current_code, current_bench = new_code, new_bench
            else:
                history_text += f"\n  refine round {round_idx+1}: {new_us:.1f} us"
                if new_us < current_bench["sim_time_us"]:
                    print(f"    top {top_idx+1} round {round_idx+1}: "
                          f"NEW BEST {new_us:.1f} us "
                          f"({100*(current_bench['sim_time_us']-new_us)/current_bench['sim_time_us']:.1f}% improvement)")
                    current_code, current_bench = new_code, new_bench
        # Record final candidate for this top.
        candidates.append({
            "name": f"refined-from-{c['name']}",
            "desc": c["desc"],
            "code": current_code,
            "bench": current_bench,
            "sim_us": current_bench["sim_time_us"],
        })

    # Pick overall best.
    if no_simulator:
        # Ablation: keep enumeration order; pick the first correct candidate.
        best = candidates[0]
        print(f"\n  [NO-SIM] Winner = {best['name']} (first valid, sim_us NOT used)")
    else:
        candidates.sort(key=lambda x: x["sim_us"])
        best = candidates[0]
    all_results.append((f"strat:{best['name'][:30]}", {
        "template": "strategy_enumerate_final",
        "cost_score": best["sim_us"] / 100.0,
        "sim_time_us": best["sim_us"],
        "local_ops": best["bench"].get("local_ops", "?"),
        "num_collective_permute": best["bench"].get("num_collective_permute", 0),
        "num_all_gather": best["bench"].get("num_all_gather", 0),
        "num_all_reduce": best["bench"].get("num_all_reduce", 0),
        "_params": {"evolved_code": best["code"]},
    }))
    all_results.sort(key=lambda x: x[1]["cost_score"])
    print(f"\n  Strategy-Enumerate complete. {len(candidates)} total candidates.")
    print(f"  Best: {best['name']} (sim={best['sim_us']:.1f} us)")
    return all_results
