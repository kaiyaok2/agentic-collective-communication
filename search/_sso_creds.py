"""Lazy SSO credential shim. Importing installs a refreshable credential
provider that shells out to `aws configure export-credentials` when boto3
needs creds. Tolerates expired/missing SSO at import time: the patch is
still installed but credential fetch only fires if/when something actually
calls AWS. With Anthropic API route in play, no AWS call happens at all.
"""
import json, subprocess, sys
from botocore.credentials import RefreshableCredentials
from botocore.session import get_session

_PROFILE = "default"

def _fetch():
    out = subprocess.check_output(
        ["aws", "configure", "export-credentials",
         "--profile", _PROFILE, "--format", "process"],
        text=True, stderr=subprocess.DEVNULL)
    d = json.loads(out)
    return {
        "access_key": d["AccessKeyId"],
        "secret_key": d["SecretAccessKey"],
        "token": d["SessionToken"],
        "expiry_time": d["Expiration"],
    }

_sess = get_session()
try:
    _meta = _fetch()
    _sess._credentials = RefreshableCredentials.create_from_metadata(
        metadata=_meta, refresh_using=_fetch, method="custom-sso-cli")
except Exception as e:
    print(f"[_sso_creds] no live SSO at import; AWS calls will fail if attempted ({e})",
          file=sys.stderr)

import boto3
boto3.DEFAULT_SESSION = boto3.session.Session(botocore_session=_sess)
