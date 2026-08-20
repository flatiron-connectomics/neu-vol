"""AWS credential bootstrap for S3 access.

Workaround for tensorstore 0.1.84: its S3 *profile* credential provider fails to
resolve ``~/.aws/credentials`` (ignores ``AWS_SHARED_CREDENTIALS_FILE`` too), but
its *environment* provider works. So when opening an S3 store we load the desired
profile ourselves and export ``AWS_*`` env vars (only if not already set). Secrets
stay in-process — never logged. The credentials file lives on shared home, so this
also works on SLURM workers (each bootstraps itself) without putting secrets into
job scripts or propagating them across the cluster.
"""

from __future__ import annotations

import configparser
import os

_KEYS = (
    ("aws_access_key_id", "AWS_ACCESS_KEY_ID"),
    ("aws_secret_access_key", "AWS_SECRET_ACCESS_KEY"),
    ("aws_session_token", "AWS_SESSION_TOKEN"),
)


def ensure_aws_credentials(profile: str | None = None, credentials_file: str | None = None) -> bool:
    """Populate AWS_* env from a credentials file profile if not already set.

    No-op (returns False) when ``AWS_ACCESS_KEY_ID`` is already in the environment
    or the file/profile is unavailable. Returns True if it set credentials.
    """
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        return False
    profile = profile or os.environ.get("AWS_PROFILE") or "default"
    path = (credentials_file or os.environ.get("AWS_SHARED_CREDENTIALS_FILE")
            or os.path.expanduser("~/.aws/credentials"))
    cp = configparser.ConfigParser()
    if not cp.read(path) or profile not in cp:
        return False
    sec = cp[profile]
    if "aws_access_key_id" not in sec or "aws_secret_access_key" not in sec:
        return False
    for file_key, env_key in _KEYS:
        if sec.get(file_key):
            os.environ[env_key] = sec[file_key]
    return True
