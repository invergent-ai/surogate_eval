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

The telemetry opt-outs are here for the same reason and with the same
urgency. ``deepteam.telemetry`` and ``deepeval.telemetry`` run their opt-out
checks at import time, and without them the import itself resolves and calls
api.ipify.org, then wires up Sentry with a live DSN and an OTLP exporter.
``surogate_eval.security.red_team`` and ``surogate_eval.eval`` set these
variables, but only once they have been imported - too late for any test
module that reaches the library by another route first.

**This file is not early enough for all of it.** ``deepeval`` registers a
pytest plugin (entry point ``deepeval -> deepeval.plugins.plugin``), so
pytest imports the package during startup, before it reads any conftest -
and the telemetry has already run by the time the line below executes. What
actually stops that is ``addopts = "-p no:deepeval"`` in ``pyproject.toml``,
which refuses to load the plugin at all. The variable below still matters
for the test modules that import deepeval themselves, later on.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("DEEPTEAM_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
