"""The status vocabulary shared by result types and the run outcome.

This lives on its own, and imports nothing, because both ends need it: the
leaf result dataclasses that stamp a status, and ``outcome.py``, which walks
them. Keeping it here leaves ``outcome.py`` free to import a result type
later without either side importing the other.
"""

#: Statuses a node can carry when it represents a failure to measure rather
#: than a measurement. Failures above the metric level (a whole evaluation
#: crashing, a benchmark blowing up, a target that never ran) produce no
#: ``scored_n``/``errored_n`` counts of their own, so without this set they
#: were invisible: ``total`` stayed 0, ``error_rate`` stayed 0.0, and the run
#: reported "completed".
FAILED_STATUSES = frozenset({
    'failed',
    'error',
    'validation_failed',
    'incompatible',
    'unhealthy',
})
