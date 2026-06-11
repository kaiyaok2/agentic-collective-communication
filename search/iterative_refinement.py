"""
Contention-Guided Iterative Synthesis (CGIS).

A multi-turn LLM refinement loop specialized for communication scheduling.
Unlike one-shot LLM proposals or blind evolutionary search, CGIS feeds the
LLM rich, interpretable contention feedback at each iteration, enabling
targeted, structural improvements.

Inspired by:
- AlphaEvolve: LLM-guided evolution, but we add decomposable feedback
- FunSearch: iterative evaluate-and-improve, but with structured topology
  traces instead of just scalar scores

Key innovation: the LLM doesn't just see "score improved from 5.3 to 4.8".
It sees: "Step 5 (distance 24) saturated link (4,8) at 6x. Distances 8 and
24 conflict on 3 shared links." This lets the LLM make surgical modifications
instead of blind mutations.

Loop:
    1. Evaluate current schedule -> produce contention diagnosis
    2. Feed diagnosis + history to LLM -> LLM proposes targeted modification
    3. Evaluate proposal -> accept if improved
    4. Also try contention-guided swaps (suggested by the analyzer)
    5. Repeat until convergence
"""

import re
from pathlib import Path

from .contention_analysis import ContentionAnalyzer
from .generate_algo import _invoke_bedrock, _parse_schedule
from .profiling import profile_schedule, format_profiling_report

REFINEMENT_PROMPT = (
    Path(__file__).parent.parent / "prompts" / "contention_feedback.md"
).read_text()


class IterativeRefinement:
    """
    Multi-turn LLM refinement for communication schedules with
    contention-guided feedback.
    """

    def __init__(self, topology, send_counts_matrix, cost_model,
                 analyzer=None, model="haiku", use_profiling=True):
        self.topo = topology
        self.send_counts = send_counts_matrix
        self.cost_model = cost_model
        self.analyzer = analyzer or ContentionAnalyzer(
            topology, send_counts_matrix)
        self.model = model
        self.world = topology.num_cores
        self.use_profiling = use_profiling

    def refine(self, initial_schedule, template="permute_ring",
               max_rounds=8, patience=3, verbose=True):
        """
        Iteratively refine a schedule using LLM + contention feedback.

        Each round: diagnose -> LLM proposes -> evaluate -> accept/reject.
        Also tries analyzer-suggested swaps as a fast local search complement.

        Returns:
            best_schedule, best_score, history
        """
        current = list(initial_schedule)
        current_score = self._evaluate(current, template)
        best = list(current)
        best_score = current_score
        history = [{"round": 0, "score": current_score, "action": "initial"}]
        stale_rounds = 0

        if verbose:
            print(f"  CGIS starting: score={current_score:.3f}")

        for round_idx in range(1, max_rounds + 1):
            improved = False

            # --- Phase A: LLM refinement with contention + profiling feedback ---
            diagnosis = self.analyzer.diagnose_schedule(current, template)
            diagnosis_text = self.analyzer.format_diagnosis(diagnosis)

            # Generate profiling report with per-step timing breakdown
            profiling_text = ""
            if self.use_profiling:
                try:
                    if template == "permute_ring":
                        prof_params = {"schedule": current}
                    elif template == "hierarchical":
                        prof_params = {"inter_schedule": current}
                    else:
                        prof_params = {"schedule": current}
                    prof_result = profile_schedule(
                        template, prof_params, self.send_counts, self.topo)
                    profiling_text = format_profiling_report(prof_result)
                except Exception:
                    pass

            prompt = self._build_prompt(
                current, current_score, diagnosis_text, history, template,
                profiling_text=profiling_text)

            try:
                response = _invoke_bedrock(
                    prompt, model=self.model, temperature=0.7, max_tokens=4096)
                proposed = self._parse_proposal(response, current)
            except Exception as e:
                if verbose:
                    print(f"  Round {round_idx}A: LLM error: {e}")
                proposed = None

            if proposed is not None:
                proposed_score = self._evaluate(proposed, template)
                if proposed_score < current_score:
                    current = proposed
                    current_score = proposed_score
                    improved = True
                    history.append({
                        "round": round_idx,
                        "score": proposed_score,
                        "accepted": True,
                        "action": "llm_refinement",
                    })
                    if verbose:
                        delta = history[-2]["score"] - proposed_score
                        print(f"  Round {round_idx}A: LLM ACCEPTED "
                              f"score={proposed_score:.3f} (-{delta:.3f})")
                else:
                    history.append({
                        "round": round_idx,
                        "score": proposed_score,
                        "accepted": False,
                        "action": "llm_refinement",
                    })

            # --- Phase B: Contention-guided swap search ---
            swaps = self.analyzer.suggest_swaps(current, diagnosis, top_k=5)
            for i, j in swaps:
                candidate = list(current)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                cand_score = self._evaluate(candidate, template)
                if cand_score < current_score:
                    current = candidate
                    current_score = cand_score
                    improved = True
                    if verbose:
                        print(f"  Round {round_idx}B: swap({i},{j}) "
                              f"score={cand_score:.3f}")
                    break

            if current_score < best_score:
                best = list(current)
                best_score = current_score

            if improved:
                stale_rounds = 0
            else:
                stale_rounds += 1
                if verbose:
                    print(f"  Round {round_idx}: no improvement "
                          f"({stale_rounds}/{patience})")

            if stale_rounds >= patience:
                if verbose:
                    print(f"  Stopping: {patience} rounds without improvement")
                break

        return best, best_score, history

    def _evaluate(self, schedule, template):
        if template == "permute_ring":
            params = {"schedule": schedule}
        elif template == "hierarchical":
            params = {"inter_schedule": schedule}
        else:
            params = {"schedule": schedule}
        score, _ = self.cost_model.evaluate_template(template, params)
        return score

    def _build_prompt(self, schedule, score, diagnosis_text, history, template,
                      profiling_text=""):
        """Build prompt with contention feedback and profiling data for the LLM."""
        history_text = ""
        for h in history[-5:]:
            status = h.get("action", "?")
            if "accepted" in h:
                status += " (ACCEPTED)" if h["accepted"] else " (rejected)"
            history_text += f"  Round {h['round']}: score={h['score']:.3f} {status}\n"

        n_elements = len(schedule)
        elements_desc = (
            f"device-level distances 1..{n_elements}"
            if template == "hierarchical"
            else f"rank-level distances 1..{n_elements}"
        )

        # Combine contention diagnosis with profiling data
        full_diagnosis = diagnosis_text
        if profiling_text:
            full_diagnosis += "\n\n" + profiling_text

        return REFINEMENT_PROMPT.replace(
            "{current_schedule}", repr(schedule)
        ).replace(
            "{current_score}", f"{score:.3f}"
        ).replace(
            "{diagnosis}", full_diagnosis
        ).replace(
            "{history}", history_text
        ).replace(
            "{template}", template
        ).replace(
            "{elements_description}", elements_desc
        ).replace(
            "{n_elements}", str(n_elements)
        )

    def _parse_proposal(self, response, current_schedule):
        """Parse LLM's proposed schedule modification from response."""
        expected = set(current_schedule)
        n = len(current_schedule)

        # Try standard schedule parse patterns
        patterns = [
            r"(?:new_|proposed_|child_)?schedule\s*=\s*\[([^\]]+)\]",
            r"\[(\d+(?:\s*,\s*\d+){%d,})\]" % (n - 2),
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, response):
                try:
                    nums = [int(x.strip()) for x in match.group(1).split(",")]
                    if set(nums) == expected and len(nums) == n:
                        return nums
                except ValueError:
                    continue

        # Parse swap instructions: "swap positions 3 and 7"
        swap_match = re.search(
            r"swap\s+(?:positions?\s+)?(\d+)\s+and\s+(\d+)", response, re.I)
        if swap_match:
            i, j = int(swap_match.group(1)), int(swap_match.group(2))
            if 0 <= i < n and 0 <= j < n:
                result = list(current_schedule)
                result[i], result[j] = result[j], result[i]
                return result

        # Parse move instructions: "move distance 8 to position 2"
        move_match = re.search(
            r"move\s+(?:distance\s+)?(\d+)\s+to\s+position\s+(\d+)",
            response, re.I)
        if move_match:
            dist = int(move_match.group(1))
            pos = int(move_match.group(2))
            if dist in expected and 0 <= pos < n:
                result = list(current_schedule)
                result.remove(dist)
                result.insert(pos, dist)
                return result

        return None
