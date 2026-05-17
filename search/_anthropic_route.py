"""
If ANTHROPIC_API_KEY is set, hot-swap:
  (1) generate_algo._invoke_bedrock -> direct Anthropic Messages API call
  (2) boto3.client("bedrock-runtime", ...) -> proxy whose .invoke_model()
      calls the Anthropic Messages API and wraps the response to look
      like a Bedrock InvokeModel response.

This covers callsites that use _invoke_bedrock (generate_algo path) and
the direct-boto3 callsites in agent_simulator_config.py (Phase 1 / refinement
tool-use loops). Anthropic Messages API and Bedrock InvokeModel share the
same wire format for messages/tools, so the proxy just forwards.
"""
import os, io, json, urllib.request, urllib.error, time

ANTHROPIC_MODELS = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-5-20250929",
    "opus":   "claude-opus-4-7",
}

def _map_model_id(model_id):
    """Map Bedrock model id to Anthropic API model id."""
    # Bedrock IDs look like "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    # or our local aliases "sonnet"/"haiku"/"opus"
    if model_id in ANTHROPIC_MODELS:
        return ANTHROPIC_MODELS[model_id]
    if model_id.startswith("us.anthropic."):
        base = model_id[len("us.anthropic."):]
        # strip "-v1:0" suffix
        if ":" in base:
            base = base.split(":", 1)[0]
        # strip the trailing "-v1" version
        if base.endswith("-v1"):
            base = base[:-3]
        return base
    return model_id


def _post_anthropic(body_dict, api_key=None, timeout=180):
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    body = json.dumps(body_dict).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    last_err = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                last_err = e
                time.sleep(min(2 ** attempt, 30))
                continue
            raise
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise last_err


def _invoke_anthropic(prompt, model="haiku", max_tokens=4096, temperature=1.0):
    out = _post_anthropic({
        "model": ANTHROPIC_MODELS.get(model, model),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    })
    return out["content"][0]["text"]


# ---------- Bedrock client proxy ----------
class _StreamLike:
    """Look like the body StreamingBody returned by Bedrock invoke_model."""
    def __init__(self, payload_dict):
        self._buf = io.BytesIO(json.dumps(payload_dict).encode("utf-8"))
    def read(self, n=-1):
        return self._buf.read(n if n != -1 else None)


class _BedrockProxy:
    """Mimics boto3 bedrock-runtime client; routes invoke_model to Anthropic."""
    def invoke_model(self, modelId=None, body=None, **kw):
        req_body = json.loads(body) if isinstance(body, (bytes, bytearray, str)) else body
        # Anthropic API expects different "anthropic_version" key in body.
        # Both Bedrock and Anthropic use the same messages/tools schema,
        # so just drop the bedrock-specific key.
        req_body.pop("anthropic_version", None)
        req_body["model"] = _map_model_id(modelId)
        payload = _post_anthropic(req_body)
        # The body field on Bedrock\s response is a stream that .read()s the JSON.
        return {"body": _StreamLike(payload)}


def install():
    """Install both hot-swaps. Idempotent."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    # Patch _invoke_bedrock
    from . import generate_algo as _ga
    _ga._invoke_bedrock = _invoke_anthropic
    # Patch boto3.client to return our proxy for bedrock-runtime
    import boto3
    if not getattr(boto3, "_anthropic_route_patched", False):
        _orig_client = boto3.client
        def _patched_client(service_name, *a, **kw):
            if service_name == "bedrock-runtime":
                return _BedrockProxy()
            return _orig_client(service_name, *a, **kw)
        boto3.client = _patched_client
        boto3._anthropic_route_patched = True
    return True


# Auto-install on import.
install()
