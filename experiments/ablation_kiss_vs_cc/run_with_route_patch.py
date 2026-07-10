"""cc-react wrapper for the Phase-3 ablation.

  (1) Rebinds `search.template_evolution._invoke_bedrock` to the direct
      Anthropic Messages API path (so the 1-node master doesn't need
      AWS Bedrock SSO credentials to run cc-react).
  (2) Forces `phase1_profiling(use_llm=False, ...)` so Phase 1 is
      deterministic and costs zero tokens -- isolating Phase 3 as the
      only variable in the ablation.
  (3) Invokes `experiments.run_search.main()` in-process so the two
      monkey-patches above stick.

Environment:
  ACC_REPO             Root of the agentic-collective-communication repo.
                       Defaults to two directories up from this file.
  ANTHROPIC_API_KEY    Required.
"""
import os
import sys

assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY missing"

HERE = os.path.dirname(os.path.abspath(__file__))
ACC = os.environ.get(
    "ACC_REPO",
    os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, ACC)

# (1) Route Bedrock calls to Anthropic API and rebind in template_evolution.
from search import _anthropic_route, template_evolution
template_evolution._invoke_bedrock = _anthropic_route._invoke_anthropic
print("[route_patch] template_evolution._invoke_bedrock -> _invoke_anthropic",
      flush=True)

# (2) Force Phase 1 into deterministic no-LLM mode.
import experiments.run_search as run_search
_orig_phase1 = run_search.phase1_profiling
def _phase1_default(use_llm, llm_model, num_nodes, verbose=True):
    return _orig_phase1(use_llm=False, llm_model=llm_model,
                        num_nodes=num_nodes, verbose=verbose)
run_search.phase1_profiling = _phase1_default
print("[route_patch] phase1_profiling forced to use_llm=False", flush=True)

# (3) Hand off argv and invoke run_search's main in-process.
script = os.path.join(ACC, "experiments", "run_search.py")
sys.argv = [script] + sys.argv[1:]
run_search.main()
