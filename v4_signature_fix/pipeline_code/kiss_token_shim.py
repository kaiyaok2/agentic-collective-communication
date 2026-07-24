"""Patches kiss to (a) log per-call token usage, (b) route through Bedrock
when USE_BEDROCK=1."""
import os, json, time
LOG = os.environ.get("ABLATION_TOKEN_LOG", "/tmp/tokens.jsonl")
TOKEN_LOG = os.environ.get("TOKEN_LOG", LOG)  # accept either env name
LOG = TOKEN_LOG

_MODEL_ID_TO_BEDROCK = {
    "claude-sonnet-4-5-20250929": "us.anthropic.claude-opus-4-8",  # sonnet-4-5 -> opus-4-8 for algo evolution
    "claude-sonnet-4-5": "us.anthropic.claude-opus-4-8",
    "claude-opus-4-1-20250820": "us.anthropic.claude-opus-4-1-20250820-v1:0",
    "claude-opus-4-1": "us.anthropic.claude-opus-4-1-20250820-v1:0",
    "claude-opus-4-5": "us.anthropic.claude-opus-4-5-20251015-v1:0",
    "claude-opus-4-7": "us.anthropic.claude-opus-4-7",
    "claude-opus-4-8": "us.anthropic.claude-opus-4-8",
    "claude-sonnet-4-6": "us.anthropic.claude-sonnet-4-6",
    "claude-sonnet-5": "us.anthropic.claude-sonnet-5",
    "claude-fable-5": "us.anthropic.claude-fable-5",
    "claude-haiku-4-5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-3-5-sonnet-latest": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-3-5-sonnet-20241022": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
}

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

USE_BEDROCK = os.environ.get("USE_BEDROCK", "").strip() in ("1", "true", "True")
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "us-east-1")


def _strip_cache_control(kwargs):
    import copy
    kwargs = copy.deepcopy(kwargs)
    def _clean(o):
        if isinstance(o, dict):
            o.pop("cache_control", None)
            for v in o.values():
                _clean(v)
        elif isinstance(o, list):
            for x in o:
                _clean(x)
    _clean(kwargs)
    # Also strip thinking key (or convert enabled→adaptive) so it works on newer models
    if "thinking" in kwargs:
        t = kwargs["thinking"]
        if isinstance(t, dict) and t.get("type") == "enabled":
            kwargs["thinking"] = {"type": "adaptive"}
    return kwargs


try:
    import kiss.core.models.anthropic_model as _amod

    if USE_BEDROCK:
        from anthropic import AnthropicBedrock

        class _BedrockShim:
            def __init__(self, *args, **kwargs):
                kwargs.pop("api_key", None)
                self._real = AnthropicBedrock(aws_region=BEDROCK_REGION)

            def __getattr__(self, name):
                if name == "messages":
                    real_msgs = self._real.messages
                    class M:
                        def create(inner, **kwargs):
                            m = kwargs.get("model", "?")
                            if m in _MODEL_ID_TO_BEDROCK:
                                kwargs["model"] = _MODEL_ID_TO_BEDROCK[m]
                            kwargs = _strip_cache_control(kwargs)
                            return real_msgs.create(**kwargs)
                        def stream(inner, **kwargs):
                            m = kwargs.get("model", "?")
                            if m in _MODEL_ID_TO_BEDROCK:
                                kwargs["model"] = _MODEL_ID_TO_BEDROCK[m]
                            kwargs = _strip_cache_control(kwargs)
                            return real_msgs.stream(**kwargs)
                        def __getattr__(inner, name2):
                            return getattr(real_msgs, name2)
                    return M()
                return getattr(self._real, name)

        _amod.Anthropic = _BedrockShim
        print(f"[shim] Anthropic -> AnthropicBedrock ({BEDROCK_REGION})",
              flush=True)

    _orig = _amod.AnthropicModel._create_message
    def _patched(self, kwargs):
        r = _orig(self, kwargs)
        try:
            _record(getattr(r, "usage", None),
                    kwargs.get("model", getattr(self, "model_name", "?")))
        except Exception:
            pass
        return r
    _amod.AnthropicModel._create_message = _patched
    print(f"[shim] kiss AnthropicModel._create_message patched -> {LOG}",
          flush=True)
except Exception as e:
    print(f"[shim] not patched: {e}", flush=True)
