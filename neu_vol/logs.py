"""Drop the object-store logging that is noise, and only that.

TensorStore's S3 stack logs at absl ERROR severity on paths that are not errors:
building the credential provider chain reports each provider that could not be
constructed *before* falling through to the one that works. A run that succeeds
completely still prints two of these per store open. There is no env var that
turns them off (checked: ``TENSORSTORE_VERBOSE_LOGGING``, ``ABSL_MIN_LOG_LEVEL``,
``AWS_CRT_LOG_LEVEL``, ``GLOG_minloglevel`` — none reduce the volume), and they
come from C++ on fd 2, so ``contextlib.redirect_stderr`` cannot see them.

So :func:`quiet_store_logs` filters fd 2 through a pipe, dropping lines that match
a short list of known-benign patterns and passing **everything else** through
untouched. Two rules keep it from hiding real problems:

1. It is a deny-list of specific known-noise strings, not a severity filter. An
   unrecognized message is always passed through.
2. :data:`NEVER_DROP` wins over every pattern, so anything mentioning denied
   access, an HTTP error code, or a tensorstore failure status is printed even if
   it also matches a noise pattern.

This is for **entry points** (scripts, drivers). Library code must not touch
process-wide stderr.

:func:`quiet_reads` is the same filter packaged for the *other* kind of entry point —
a library function that a notebook calls directly, where there is no ``main()`` to wrap.
It is still opt-in (the consumer's own function chooses to use it), so the rule above
holds; what it removes is each consumer reimplementing the depth counter that makes
nesting and threading safe. neu-draw arrived at that pattern first and its reasoning is
worth repeating: expecting a user to run an incantation before their output is legible is
the worse trade, because forgetting it after a kernel restart looks exactly like the
filter being broken.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
import threading

# Emitted on every successful S3 open: the credential providers that could not be
# built, before the environment provider (which aws.ensure_aws_credentials
# populates) answers. See CLAUDE.md invariant 8.
BENIGN_PATTERNS = (
    "Profile credentials provider could not load a profile",
    "Failed to resolve either region, role arn or token file path",
    "sts web identity provider initialization",
    # Reached only when no credentials are in the environment; the chain probes
    # the EC2 metadata service, which does not exist off EC2, and times out.
    "AWS_IO_SOCKET_TIMEOUT",
    "AWS_IO_SOCKET_NO_ROUTE_TO_HOST",
    "IMDS Client failed to acquire a connection",
    "IMDS client failed to update the token",
    "Failed to obtain new connection from http layer",
    "Failed to complete connection acquisition",
    "Client connection failed with error",
    "timed out, shutting down.",
    # The GCS equivalent, emitted on every anonymous read of a public bucket: the
    # provider chain reports that no gcloud application-default credentials exist
    # before falling through to anonymous access, which then succeeds. Matching the
    # source file catches its siblings too, and is safe because a *real* GCS auth
    # failure says PERMISSION_DENIED or UNAUTHENTICATED, which NEVER_DROP keeps.
    "google_auth_provider.cc",
    "Could not find the credentials file in the standard gcloud location",
)

# Printed no matter what. A message that mentions any of these is diagnostic even
# if it also looks like noise.
NEVER_DROP = (
    "PERMISSION_DENIED", "AccessDenied", "Access Denied", "UNAUTHENTICATED",
    "NOT_FOUND", "INVALID_ARGUMENT", "ALREADY_EXISTS", "FAILED_PRECONDITION",
    "Traceback", "http_response_code",
)

# absl's prefix: severity, date, time, thread id. Only such lines are candidates;
# ordinary output is never examined for noise patterns.
_ABSL = re.compile(rb"^[EWIF]\d{4} \d")


def is_benign(line: bytes) -> bool:
    """True if this line is known store-stack noise and safe to drop."""
    if not _ABSL.match(line):
        return False
    text = line.decode("utf-8", "replace")
    if any(m in text for m in NEVER_DROP):
        return False
    return any(p in text for p in BENIGN_PATTERNS)


@contextlib.contextmanager
def quiet_store_logs(enabled: bool = True):
    """Filter benign object-store logging out of fd 2 for the duration.

    No-op when ``enabled`` is false, or when fd 2 cannot be duplicated (a closed
    or exotic stderr) — in that case everything is passed through unfiltered,
    which is the safe direction.
    """
    if not enabled:
        yield
        return
    try:
        saved = os.dup(2)
    except OSError:
        yield
        return

    read_fd, write_fd = os.pipe()

    def pump():
        with os.fdopen(read_fd, "rb", 0) as r:
            for line in r:
                if not is_benign(line):
                    os.write(saved, line)

    thread = threading.Thread(target=pump, name="stderr-filter", daemon=True)
    thread.start()
    try:
        sys.stderr.flush()
        os.dup2(write_fd, 2)
        os.close(write_fd)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)          # restore before draining, so nothing is lost
        thread.join(timeout=5.0)
        os.close(saved)


# --------------------------------------------------------------------------- #
# the notebook-facing wrapper
# --------------------------------------------------------------------------- #
#: Set False to see raw store logging, including the benign lines. Consulted at call
#: time, so flipping it in a live session takes effect on the next read.
reads_quiet = True

_scoped = None
_depth = 0
_depth_lock = threading.RLock()


@contextlib.contextmanager
def quiet_reads():
    """Filter benign store logging for one read, safe to nest and to call from threads.

    For a library function that a notebook calls directly — there is no ``main()`` to
    wrap, and from the notebook's point of view that function *is* the entry point. Wrap
    the store-touching call in it:

        with quiet_reads():
            info = location.read_json(source, "info")

    A no-op when :data:`reads_quiet` is false or when an outer read is already
    filtering. The depth counter matters: only the thread that got there first ever
    swaps fd 2, because two threads racing ``dup2`` on the same descriptor is a way to
    lose output rather than merely filter it.
    """
    global _depth, _scoped
    if not reads_quiet:
        yield
        return
    with _depth_lock:
        first = _depth == 0
        if first:
            _scoped = quiet_store_logs(True)
            _scoped.__enter__()
        _depth += 1
    try:
        yield
    finally:
        with _depth_lock:
            _depth -= 1
            if _depth == 0 and _scoped is not None:
                manager, _scoped = _scoped, None
                manager.__exit__(None, None, None)
