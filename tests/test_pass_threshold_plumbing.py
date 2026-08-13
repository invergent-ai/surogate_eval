"""`pass_threshold` survives every hop from the YAML to the backend.

The rule itself is covered in `test_pass_threshold.py`, which calls
`row_passed` directly. That is the wrong place to find out whether the
setting arrives: it shipped complete and correct and did nothing at all,
because three separate layers rebuild the config from explicit whitelists
and none of them carried the key --

    bench entry dict  ->  runners._run_single_benchmark  (explicit kwargs)
                      ->  BenchmarkConfig               (dataclass fields)
                      ->  generic.evaluate              (explicit dict keys)
                      ->  backend `config`

so `config.get('pass_threshold')` was always None and every judge row still
passed on `score > 0`. This is the failure ops' own `_custom_eval_entry`
docstring warns about: "A key this function does not copy is a key the
runner never sees, and that failure is invisible from the outside -- the
API accepts the config and the run completes, just scored by a rule nobody
chose."

These tests drive the real hops rather than asserting the field exists.
"""

from surogate_eval.benchmarks import BenchmarkConfig
from surogate_eval.benchmarks.generic import GenericBenchmark
from surogate_eval.targets.base import TargetType


class FakeTarget:
    target_type = TargetType.LLM
    name = "t1"
    config = {"model": "m", "base_url": "https://target.example", "api_key": "k"}


class CapturingBackend:
    """Records the config the benchmark hands its backend."""

    def __init__(self):
        self.config = None

    def evaluate(self, target, benchmark_name, config):
        self.config = config
        return {"overall_score": 0.0, "task_results": {}, "detailed_results": []}


def _config_handed_to_backend(bench_config: dict) -> dict:
    """Drive the real path: entry dict -> BenchmarkConfig -> backend config.

    Mirrors what `runners._run_single_benchmark` builds, using the same
    field names, so a kwarg dropped there shows up here.
    """
    config = BenchmarkConfig(
        name=bench_config.get("name"),
        backend=bench_config.get("backend", "evalscope"),
        limit=bench_config.get("limit"),
        pass_threshold=bench_config.get("pass_threshold"),
    )
    benchmark = GenericBenchmark(config)
    capturing = CapturingBackend()
    benchmark.backend = capturing
    benchmark.evaluate(FakeTarget())
    return capturing.config


def test_the_threshold_reaches_the_backend():
    handed = _config_handed_to_backend(
        {"name": "mt_bench", "pass_threshold": 0.8, "limit": 10},
    )
    assert handed["pass_threshold"] == 0.8


def test_an_absent_threshold_reads_as_no_threshold():
    """`generic.evaluate` runs `remove_none_values` before handing the
    config over, so an unset threshold arrives as a *missing key* rather
    than an explicit None. Both read the same through
    `config.get('pass_threshold')`, which is how both backends fetch it --
    what must not happen is it arriving as some other falsy value that would
    read as a real threshold of 0.
    """
    handed = _config_handed_to_backend({"name": "mt_bench", "limit": 10})
    assert handed.get("pass_threshold") is None


def test_a_zero_threshold_survives_the_trip():
    """0.0 is falsy. Any hop using `or` or a truthiness guard would drop it
    to None, which is a different rule: at-least-zero passes a zero row, the
    legacy evalscope rule does not.
    """
    handed = _config_handed_to_backend(
        {"name": "mt_bench", "pass_threshold": 0.0},
    )
    assert handed["pass_threshold"] == 0.0


def test_runners_passes_the_key_into_benchmarkconfig():
    """The hop the dataclass cannot show on its own.

    `runners._run_single_benchmark` names every field explicitly, so a new
    one is carried only if someone adds a line there. Reading the source is
    the honest check: constructing a BenchmarkConfig here would pass even
    with that line deleted.
    """
    import inspect

    from surogate_eval import runners

    src = inspect.getsource(runners._run_single_benchmark)
    assert 'pass_threshold=bench_config.get("pass_threshold")' in src, (
        "runners._run_single_benchmark must carry pass_threshold into "
        "BenchmarkConfig, or the setting silently does nothing"
    )
