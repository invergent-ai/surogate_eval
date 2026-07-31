"""Session-wide test setup.

``HF_HUB_OFFLINE``/``HF_DATASETS_OFFLINE`` must be set here, at conftest
import time, rather than inside a fixture or a test body. ``huggingface_hub``
snapshots these into module-level constants (``huggingface_hub.constants``)
the first time it is imported, and ``datasets.load_dataset`` consults those
constants rather than re-reading the environment. If the library is already
imported by the time a test sets the variable, the setting is a no-op for the
rest of the process. Setting it here is early enough: this file is the first
thing pytest imports under ``tests/``, before any test module - and before
this repo's own code - has had a chance to import ``datasets`` or
``huggingface_hub``.

This keeps ``datasets.load_dataset('csv', ...)`` (used by the benchmark tests
in ``tests/test_run_exit_code.py`` over a local, on-disk CSV) from doing a
Hub lookup before falling back to the local file. No test may make a network
call.

``DEEPTEAM_TELEMETRY_OPT_OUT`` is here for the same reason and with the same
urgency. ``deepteam.telemetry`` runs its opt-out check at import time, and
without it the import itself resolves and calls api.ipify.org, then wires up
Sentry and an OTLP exporter. ``surogate_eval.security.red_team`` sets the
variable, but only once it has been imported - which is too late for any
test module that reaches deepteam by another route first.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("DEEPTEAM_TELEMETRY_OPT_OUT", "YES")
