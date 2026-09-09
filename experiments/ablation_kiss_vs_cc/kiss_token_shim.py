"""Patches kiss.core.models.anthropic_model.AnthropicModel._create_message to
log per-call token usage to ABLATION_TOKEN_LOG."""
import os, json, time
LOG = os.environ.get("ABLATION_TOKEN_LOG", "/tmp/tokens.jsonl")

def _record(usage_obj, model_id):
    try:
        rec = {
            "ts": time.time(),
            "model": model_id,
            "input_tokens": int(getattr(usage_obj, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage_obj, "output_tokens", 0) or 0),
            "cache_creation_input_tokens": int(getattr(usage_obj, "cache_creation_input_tokens", 0) or 0),
            "cache_read_input_tokens": int(getattr(usage_obj, "cache_read_input_tokens", 0) or 0),
        }
        os.makedirs(os.path.dirname(LOG) or "/tmp", exist_ok=True)
        with open(LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass

try:
    from kiss.core.models.anthropic_model import AnthropicModel
    _orig = AnthropicModel._create_message
    def _patched(self, kwargs):
        r = _orig(self, kwargs)
        try:
            _record(getattr(r, "usage", None),
                    kwargs.get("model", getattr(self, "model_name", "?")))
        except Exception:
            pass
        return r
    AnthropicModel._create_message = _patched
    print(f"[shim] kiss AnthropicModel._create_message patched -> {LOG}",
          flush=True)
except Exception as e:
    print(f"[shim] not patched: {e}", flush=True)
