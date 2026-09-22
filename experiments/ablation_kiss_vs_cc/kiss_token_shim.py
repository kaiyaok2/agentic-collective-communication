"""Patches kiss.core.models.anthropic_model.AnthropicModel._create_message to
log per-call token usage to ABLATION_TOKEN_LOG.

When CLAUDE_CODE_USE_BEDROCK is set, ALSO swaps the kiss library's direct
Anthropic client for an AnthropicBedrock client (session AWS creds), so the kiss
pipeline sidesteps the direct-API workspace usage cap exactly like the overlay
path. The swap maps the Anthropic model id -> Bedrock inference-profile id and
strips cache_control blocks (Bedrock invoke rejects them). See memory
[[kiss_bedrock_shim_fix]]."""
import os, json, time
LOG = os.environ.get("ABLATION_TOKEN_LOG", "/tmp/tokens.jsonl")


def _anthropic_to_bedrock_id(model_id):
    if not model_id or model_id.startswith("us.anthropic."):
        return model_id
    known = {
        "claude-sonnet-4-5-20250929": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "claude-haiku-4-5-20251001":  "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "claude-opus-4-7":            "us.anthropic.claude-opus-4-7-v1:0",
    }
    return known.get(model_id, f"us.anthropic.{model_id}-v1:0")


def _strip_cache_control(obj):
    """Recursively drop cache_control keys (Bedrock invoke rejects them)."""
    if isinstance(obj, dict):
        return {k: _strip_cache_control(v) for k, v in obj.items()
                if k != "cache_control"}
    if isinstance(obj, list):
        return [_strip_cache_control(v) for v in obj]
    return obj


def _install_bedrock_client():
    """Force AnthropicModel to build an AnthropicBedrock client instead of the
    direct Anthropic client, and adapt kwargs (model id + cache_control) on each
    streamed message. Idempotent."""
    import anthropic
    from kiss.core.models import anthropic_model as _am

    region = (os.environ.get("AWS_REGION")
              or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1")

    def _adapt(kwargs):
        kwargs = dict(kwargs)
        kwargs["model"] = _anthropic_to_bedrock_id(kwargs.get("model"))
        # kiss sets a TOP-LEVEL cache_control kwarg (enable_cache); Bedrock
        # rejects it anywhere in the body, so drop it at every level.
        kwargs.pop("cache_control", None)
        for key in ("system", "messages", "tools"):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = _strip_cache_control(kwargs[key])
        return kwargs

    class _BedrockMessages:
        def __init__(self, inner):
            self._inner = inner
        def stream(self, **kwargs):
            return self._inner.messages.stream(**_adapt(kwargs))
        def create(self, **kwargs):
            return self._inner.messages.create(**_adapt(kwargs))
        def __getattr__(self, name):
            return getattr(self._inner.messages, name)

    class _BedrockClient:
        def __init__(self, inner=None):
            self._c = inner or anthropic.AnthropicBedrock(aws_region=region)
            self.messages = _BedrockMessages(self._c)
        def with_options(self, **kw):
            # keep the kwargs adaptation on clients derived via with_options
            # (kiss calls client.with_options(max_retries=0).messages.create)
            return _BedrockClient(self._c.with_options(**kw))
        def __getattr__(self, name):
            return getattr(self._c, name)

    # After initialize() builds its direct-API client, swap in the Bedrock one.
    _orig_initialize = _am.AnthropicModel.initialize

    def _init(self, *a, **kw):
        _orig_initialize(self, *a, **kw)
        self.client = _BedrockClient()
        self._client_inputs = ("bedrock", "bedrock")

    _am.AnthropicModel.initialize = _init
    print(f"[shim] kiss client -> AnthropicBedrock ({region})", flush=True)

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

if os.environ.get("CLAUDE_CODE_USE_BEDROCK"):
    try:
        _install_bedrock_client()
    except Exception as e:
        print(f"[shim] bedrock client NOT installed: {e}", flush=True)
