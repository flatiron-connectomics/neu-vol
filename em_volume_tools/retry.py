"""Bounded retry for transient object-store failures.

Object stores fail transiently at a rate that only shows up at scale: a
whole-volume copy is tens of thousands of requests, and TLS connections get reset,
throttled, or 500'd. TensorStore retries at the request level, but errors still
surface (observed: ``UNAVAILABLE ... CURL error SSL connect error: Recv failure:
Connection reset by peer``, ``curl_code=35``, which killed a 10,692-block copy
after it had already succeeded).

**Classification is by message text, not exception type.** TensorStore maps absl
status codes onto builtin Python exceptions, so both ``PERMISSION_DENIED`` and
``UNAVAILABLE`` arrive as ``ValueError`` — the type carries no signal. Permanent
markers are therefore checked *first* and win: a 403 must never be retried, or a
misconfigured run burns its whole backoff budget on every task before failing.
Unrecognized errors are treated as **permanent**, so a new failure mode surfaces
promptly instead of being retried into a timeout.

Only wrap operations that are safe to repeat. Writing a block of voxels to the
same region is idempotent; appending to a manifest is not.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Checked FIRST — these will not fix themselves, so retrying only wastes time.
PERMANENT_MARKERS = (
    "PERMISSION_DENIED",     # 403: credentials/policy. See aws.ensure_aws_credentials.
    "NOT_FOUND",             # missing object/prefix
    "INVALID_ARGUMENT",      # malformed spec or region
    "FAILED_PRECONDITION",
    "OUT_OF_RANGE",
    "ALREADY_EXISTS",
    "UNAUTHENTICATED",
)

# Retryable: network, throttling, and server-side faults.
TRANSIENT_MARKERS = (
    "UNAVAILABLE",           # tensorstore's bucket for connection failures
    "DEADLINE_EXCEEDED",
    "RESOURCE_EXHAUSTED",    # throttling
    "ABORTED",
    "CURL error",            # transport-level, e.g. curl_code=35
    "Connection reset",
    "Connection refused",
    "Broken pipe",
    "SlowDown",              # S3 throttling
    "InternalError",         # S3 5xx
    "ServiceUnavailable",
    "RequestTimeout",
    "http_response_code='50",  # 500/503
)


def is_transient(exc: BaseException) -> bool:
    """Would retrying this error plausibly succeed?

    Permanent markers take precedence, and an unrecognized error is permanent.
    """
    msg = str(exc)
    if any(m in msg for m in PERMANENT_MARKERS):
        return False
    return any(m in msg for m in TRANSIENT_MARKERS)


def with_retry(fn: Callable[[], T], *, attempts: int = 5, base_delay: float = 1.0,
               max_delay: float = 30.0, label: str = "", jitter: bool = True) -> T:
    """Call ``fn``, retrying transient failures with exponential backoff.

    Raises the last exception once ``attempts`` is exhausted, and re-raises a
    permanent error immediately. ``jitter`` spreads retries across concurrent
    workers — without it, 48 workers that hit the same throttle all come back
    simultaneously and reproduce it.
    """
    delay = base_delay
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:                    # noqa: BLE001 - classified below
            if not is_transient(exc) or attempt == attempts:
                raise
            wait = delay * (random.uniform(0.5, 1.5) if jitter else 1.0)
            logger.warning("transient failure%s (attempt %d/%d), retrying in %.1fs: %s",
                           f" on {label}" if label else "", attempt, attempts, wait,
                           str(exc)[:200])
            time.sleep(wait)
            delay = min(delay * 2, max_delay)
    raise AssertionError("unreachable")             # pragma: no cover
