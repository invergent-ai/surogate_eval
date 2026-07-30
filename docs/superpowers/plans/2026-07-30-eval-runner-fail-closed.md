# Eval Runner Fail-Closed Implementation Plan

> **How to use this plan:** work through the tasks in order; each one is self-contained and ends in a commit. Steps use checkbox (`- [ ]`) syntax so progress can be tracked in place.

**Goal:** Make the eval runner report failure instead of fabricating scores, so a broken judge, a missing credential, or an unreachable target produces an error rather than a confident-looking zero.

**Architecture:** One uniform mechanism. A typed error is raised at the source (wrapper, config loader, health check), caught at the `MetricResult` boundary (adapter, safety metrics), carried through aggregation as an `errored` status excluded from averages, and rolled up into a run-level outcome that sets the process exit code. No ledger or shared mutable state: every path already funnels through `MetricResult`.

**Tech Stack:** Python 3.12+, pydantic 2, deepeval 3.7.9, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-07-30-eval-runner-fail-closed-design.md`

## Global Constraints

- Repo: `surogate_eval`. Branch: `fix/eval-runner-fail-closed` (already created off `main`).
- Python `>=3.12,<3.14`.
- `max_error_rate` default is **0.2**. Configurable as a top-level eval config key.
- Errored results carry `score=None`, never `0.0`.
- No test may make a network call. Use fake targets and fake HTTP clients throughout.
- Tests requiring deepeval need the `security` extra: `uv sync --extra security --extra test`.
- Commit after every task. Conventional commit format: `type(scope): imperative summary`.
- Commit messages describe the change and nothing else: no tooling notes, no attribution trailers.

---

### Task 1: Test infrastructure and error taxonomy

**Files:**
- Modify: `pyproject.toml` (add `test` extra)
- Create: `surogate_eval/errors.py`
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_errors.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `EvalError`, `ConfigError`, `TargetUnhealthyError`, `JudgeError`, `JudgeUnavailableError`, `JudgeParseError`. All later tasks import from `surogate_eval.errors`.

- [ ] **Step 1: Add the test extra to `pyproject.toml`**

In the `[project.optional-dependencies]` block, after the existing `pdf` entry, add:

```toml
test = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
]
```

- [ ] **Step 2: Install the test environment**

Run: `uv sync --extra security --extra test`

Expected: succeeds, `deepeval==3.7.9` and `pytest` both present.

- [ ] **Step 3: Write the failing test**

Create `tests/__init__.py` as an empty file, then create `tests/test_errors.py`:

```python
import pytest

from surogate_eval.errors import (
    ConfigError,
    EvalError,
    JudgeError,
    JudgeParseError,
    JudgeUnavailableError,
    TargetUnhealthyError,
)


@pytest.mark.parametrize(
    "cls",
    [ConfigError, TargetUnhealthyError, JudgeError,
     JudgeUnavailableError, JudgeParseError],
)
def test_every_error_is_an_eval_error(cls):
    assert issubclass(cls, EvalError)


@pytest.mark.parametrize("cls", [JudgeUnavailableError, JudgeParseError])
def test_judge_errors_share_a_base(cls):
    """Catch sites catch JudgeError to handle both judge failure modes."""
    assert issubclass(cls, JudgeError)


def test_config_error_is_not_a_judge_error():
    """A bad config must not be swallowed by judge-failure handling."""
    assert not issubclass(ConfigError, JudgeError)
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `uv run --extra security --extra test pytest tests/test_errors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'surogate_eval.errors'`

- [ ] **Step 5: Write the implementation**

Create `surogate_eval/errors.py`:

```python
"""Typed errors for the eval runner.

The runner must fail closed: a broken judge, an unreachable target or an
unresolved config variable has to surface as an error, never as a score.
These types let each catch site tell those cases apart from a genuine bug
in our own code.
"""


class EvalError(Exception):
    """Base for every error the eval runner raises deliberately."""


class ConfigError(EvalError):
    """Config is unusable. Raised at load, before any evaluation runs."""


class TargetUnhealthyError(EvalError):
    """A target failed its health check and cannot be evaluated."""


class JudgeError(EvalError):
    """Base for judge failures. Catch this to handle both modes below."""


class JudgeUnavailableError(JudgeError):
    """The judge could not be reached, or returned no content."""


class JudgeParseError(JudgeError):
    """The judge returned content that could not be parsed into a schema."""
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run --extra security --extra test pytest tests/test_errors.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml surogate_eval/errors.py tests/__init__.py tests/test_errors.py
git commit -m "test(eval): bootstrap pytest and add the eval error taxonomy"
```

---

### Task 2: Errored status on MetricResult and honest aggregation

**Files:**
- Modify: `surogate_eval/metrics/base.py:36-92`
- Create: `tests/test_metric_result.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `MetricStatus` enum with members `scored` and `errored`.
  - `MetricResult.status: MetricStatus` field, default `MetricStatus.scored`.
  - `MetricResult.score: Optional[float]`.
  - `MetricResult.errored(*, metric_name, metric_type, reason, metadata=None) -> MetricResult` classmethod. Tasks 6, 7 and 8 construct errored results only through this.
  - `BatchMetricResult.scored_n`, `.errored_n`, `.error_rate` properties.

- [ ] **Step 1: Write the failing test**

Create `tests/test_metric_result.py`:

```python
from surogate_eval.metrics.base import (
    BatchMetricResult,
    MetricResult,
    MetricStatus,
    MetricType,
)


def scored(value: float, success: bool = True) -> MetricResult:
    return MetricResult(
        metric_name="m",
        metric_type=MetricType.TOXICITY,
        score=value,
        success=success,
    )


def errored() -> MetricResult:
    return MetricResult.errored(
        metric_name="m",
        metric_type=MetricType.TOXICITY,
        reason="judge exploded",
    )


def test_result_defaults_to_scored():
    assert scored(1.0).status is MetricStatus.scored


def test_errored_result_has_no_score():
    """None, not 0.0, so an error can never be averaged in by accident."""
    r = errored()
    assert r.status is MetricStatus.errored
    assert r.score is None
    assert r.success is False


def test_to_dict_carries_status():
    assert scored(1.0).to_dict()["status"] == "scored"
    assert errored().to_dict()["status"] == "errored"


def test_avg_score_excludes_errored():
    """A judge outage must not drag the score down (E-RUN-1)."""
    batch = BatchMetricResult(
        metric_name="m",
        metric_type=MetricType.TOXICITY,
        results=[scored(1.0), scored(0.6), errored(), errored()],
    )
    assert batch.avg_score == 0.8
    assert batch.scored_n == 2
    assert batch.errored_n == 2
    assert batch.error_rate == 0.5


def test_success_rate_excludes_errored():
    batch = BatchMetricResult(
        metric_name="m",
        metric_type=MetricType.TOXICITY,
        results=[scored(1.0, True), scored(0.0, False), errored()],
    )
    assert batch.success_rate == 0.5


def test_all_errored_batch_does_not_divide_by_zero():
    batch = BatchMetricResult(
        metric_name="m",
        metric_type=MetricType.TOXICITY,
        results=[errored(), errored()],
    )
    assert batch.avg_score == 0.0
    assert batch.success_rate == 0.0
    assert batch.error_rate == 1.0


def test_empty_batch_has_zero_error_rate():
    batch = BatchMetricResult(
        metric_name="m", metric_type=MetricType.TOXICITY, results=[],
    )
    assert batch.error_rate == 0.0


def test_batch_to_dict_reports_counts():
    batch = BatchMetricResult(
        metric_name="m",
        metric_type=MetricType.TOXICITY,
        results=[scored(1.0), errored()],
    )
    d = batch.to_dict()
    assert d["scored_n"] == 1
    assert d["errored_n"] == 1
    assert d["error_rate"] == 0.5
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra security --extra test pytest tests/test_metric_result.py -v`
Expected: FAIL with `ImportError: cannot import name 'MetricStatus'`

- [ ] **Step 3: Write the implementation**

In `surogate_eval/metrics/base.py`, add `Enum` to the imports if absent (`from enum import Enum`) and ensure `Optional` is imported from `typing`.

Add above the `MetricResult` dataclass:

```python
class MetricStatus(str, Enum):
    """Whether a result is a measurement or a failure to measure."""

    scored = "scored"
    errored = "errored"
```

Replace the `MetricResult` dataclass body (currently lines 39-58) with:

```python
class MetricResult:
    """Result from a metric evaluation."""

    metric_name: str
    metric_type: MetricType
    score: Optional[float]
    success: bool
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: MetricStatus = MetricStatus.scored

    @classmethod
    def errored(
            cls,
            *,
            metric_name: str,
            metric_type: MetricType,
            reason: str,
            metadata: Optional[Dict[str, Any]] = None,
    ) -> "MetricResult":
        """Build a result that records a failure to measure.

        ``score`` is None rather than 0.0 so an error can never be
        averaged into a score (E-RUN-1).
        """
        return cls(
            metric_name=metric_name,
            metric_type=metric_type,
            score=None,
            success=False,
            reason=reason,
            metadata=metadata or {},
            status=MetricStatus.errored,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'metric_name': self.metric_name,
            'metric_type': self.metric_type.value,
            'score': self.score,
            'success': self.success,
            'status': self.status.value,
            'reason': self.reason,
            'metadata': self.metadata
        }
```

Replace the `BatchMetricResult` properties and `to_dict` (currently lines 67-91) with:

```python
    @property
    def scored_results(self) -> List[MetricResult]:
        """Results that are a measurement rather than a failure."""
        return [r for r in self.results if r.status is MetricStatus.scored]

    @property
    def scored_n(self) -> int:
        return len(self.scored_results)

    @property
    def errored_n(self) -> int:
        return len(self.results) - self.scored_n

    @property
    def error_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.errored_n / len(self.results)

    @property
    def avg_score(self) -> float:
        """Average over what we could actually measure."""
        scored = self.scored_results
        if not scored:
            return 0.0
        return sum(r.score for r in scored) / len(scored)

    @property
    def success_rate(self) -> float:
        scored = self.scored_results
        if not scored:
            return 0.0
        return sum(1 for r in scored if r.success) / len(scored)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'metric_name': self.metric_name,
            'metric_type': self.metric_type.value,
            'num_evaluations': len(self.results),
            'scored_n': self.scored_n,
            'errored_n': self.errored_n,
            'error_rate': self.error_rate,
            'avg_score': self.avg_score,
            'success_rate': self.success_rate,
            'results': [r.to_dict() for r in self.results]
        }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --extra security --extra test pytest tests/test_metric_result.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Verify no consumer does arithmetic on a now-nullable score**

Run: `grep -rn "\.score" --include=*.py surogate_eval/ | grep -v "toxicity_score\|bias_score\|harm_score\|safety_score\|fairness_score"`

Check each hit. Two are already known safe and need no change:

- `runners.py:191` serialises `individual_result.score` straight into a dict. `None` becomes JSON
  `null`. Fine.
- `custom_eval_backend.py:571-577` reads `metric.score` from a **deepeval** metric object, not our
  `MetricResult`, so it is unaffected by this change.

If any *new* hit performs arithmetic or comparison on a `MetricResult.score`, add a `None` guard
before proceeding.

- [ ] **Step 6: Commit**

```bash
git add surogate_eval/metrics/base.py tests/test_metric_result.py
git commit -m "feat(eval): add errored metric status and exclude it from averages"
```

---

### Task 3: Config loader fails hard on unresolved variables

**Files:**
- Modify: `surogate_eval/config/loader.py:15-61`
- Create: `tests/test_config_loader.py`

**Interfaces:**
- Consumes: `ConfigError` from Task 1.
- Produces: `load_config` raises `ConfigError` listing every unresolved variable.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config_loader.py`:

```python
import pytest

from surogate_eval.config.eval_config import EvalConfig
from surogate_eval.config.loader import load_config
from surogate_eval.errors import ConfigError

CONFIG = """\
project:
  name: test
targets:
  - name: t1
    type: llm
    provider: openai
    model: gpt-4
    api_key: ${SPIKE_MISSING_KEY}
    judge_key: ${SPIKE_OTHER_MISSING}
"""


def write(tmp_path, text):
    path = tmp_path / "eval.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_unresolved_var_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("SPIKE_MISSING_KEY", raising=False)
    monkeypatch.delenv("SPIKE_OTHER_MISSING", raising=False)
    with pytest.raises(ConfigError):
        load_config(EvalConfig, write(tmp_path, CONFIG))


def test_error_lists_every_missing_var(tmp_path, monkeypatch):
    """One run per typo is a bad loop. Report them all at once."""
    monkeypatch.delenv("SPIKE_MISSING_KEY", raising=False)
    monkeypatch.delenv("SPIKE_OTHER_MISSING", raising=False)
    with pytest.raises(ConfigError) as exc:
        load_config(EvalConfig, write(tmp_path, CONFIG))
    message = str(exc.value)
    assert "SPIKE_MISSING_KEY" in message
    assert "SPIKE_OTHER_MISSING" in message


def test_resolved_vars_load_cleanly(tmp_path, monkeypatch):
    monkeypatch.setenv("SPIKE_MISSING_KEY", "sk-real")
    monkeypatch.setenv("SPIKE_OTHER_MISSING", "sk-other")
    config = load_config(EvalConfig, write(tmp_path, CONFIG))
    assert config.targets[0].api_key == "sk-real"


def test_config_without_vars_is_unaffected(tmp_path):
    text = CONFIG.replace("${SPIKE_MISSING_KEY}", "sk-literal").replace(
        "${SPIKE_OTHER_MISSING}", "sk-literal2"
    )
    config = load_config(EvalConfig, write(tmp_path, text))
    assert config.targets[0].api_key == "sk-literal"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra security --extra test pytest tests/test_config_loader.py -v`
Expected: `test_unresolved_var_raises` and `test_error_lists_every_missing_var` FAIL (no exception raised).

- [ ] **Step 3: Write the implementation**

In `surogate_eval/config/loader.py`, add the import:

```python
from surogate_eval.errors import ConfigError
```

Replace the body of `load_config` up to the `config = config_cls(cfg)` line with:

```python
def load_config(config_cls: SurogateConfig, path: str) -> SurogateConfig:
    with open(path, encoding="utf-8") as file:
        cfg_dict = yaml.safe_load(file)

        # Expand environment variables. An unresolved ${VAR} used to be
        # left in place as a literal, which then passed the target health
        # check as a non-empty credential and 401'd on every request
        # (E-RUN-2). Fail here instead, and report every missing name at
        # once so a user fixes them in one pass.
        missing: set[str] = set()
        cfg_dict = _expand_env_vars(cfg_dict, missing)
        if missing:
            raise ConfigError(
                f"Unresolved environment variable(s) in {path}: "
                f"{', '.join(sorted(missing))}. "
                "Export them, or remove the ${...} reference."
            )
        cfg: DictDefault = DictDefault(cfg_dict)
```

Replace `_expand_env_vars` entirely with:

```python
def _expand_env_vars(obj: Any, missing: set[str]) -> Any:
    """
    Recursively expand environment variables in config.
    Supports ${VAR_NAME} syntax.

    Names that cannot be resolved are collected into *missing* and left
    untouched; the caller raises once it has the full set.
    """
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v, missing) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item, missing) for item in obj]
    elif isinstance(obj, str):
        # Match ${VAR_NAME} pattern
        pattern = r'\$\{([^}]+)\}'

        def replace_env_var(match):
            var_name = match.group(1)
            value = os.environ.get(var_name)
            if value is None:
                missing.add(var_name)
                return match.group(0)
            logger.debug(f"Expanded ${{{var_name}}} (length: {len(value)})")
            return value

        return re.sub(pattern, replace_env_var, obj)
    else:
        return obj
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --extra security --extra test pytest tests/test_config_loader.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add surogate_eval/config/loader.py tests/test_config_loader.py
git commit -m "fix(eval): fail config load on unresolved environment variables"
```

---

### Task 4: Health check stops assuming healthy

**Files:**
- Modify: `surogate_eval/targets/model.py:218-235`
- Create: `tests/test_health_check.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ModelTarget.health_check()` returns False for a missing credential, a rejected credential, and an unverifiable endpoint.

- [ ] **Step 1: Write the failing test**

Create `tests/test_health_check.py`:

```python
from types import SimpleNamespace

import pytest

from surogate_eval.targets.model import ModelTarget


class FakeClient:
    """Stands in for the httpx client. Never touches the network."""

    def __init__(self, status_code=None, raises=False):
        self.status_code = status_code
        self.raises = raises
        self.calls = []

    def get(self, path, timeout=None):
        self.calls.append(path)
        if self.raises:
            raise ConnectionError("unreachable")
        return SimpleNamespace(status_code=self.status_code)

    def close(self):
        pass


def make_target(api_key, client):
    """Build a ModelTarget without running __init__ (which opens a socket)."""
    target = ModelTarget.__new__(ModelTarget)
    target.name = "t1"
    target.base_url = "https://api.openai.com/v1"
    target.api_key = api_key
    target.provider = None
    target.client = client
    return target


def test_missing_key_is_unhealthy():
    target = make_target("", FakeClient(status_code=200))
    assert target.health_check() is False


def test_rejected_credential_is_unhealthy():
    """401 is exactly the E-RUN-2 case: a key that is present but wrong."""
    target = make_target("sk-wrong", FakeClient(status_code=401))
    assert target.health_check() is False


def test_unreachable_endpoint_is_unhealthy():
    """Previously this returned True whenever a key was present."""
    target = make_target("sk-real", FakeClient(raises=True))
    assert target.health_check() is False


def test_working_endpoint_is_healthy():
    target = make_target("sk-real", FakeClient(status_code=200))
    assert target.health_check() is True


def test_placeholder_key_is_unhealthy():
    """Belt and braces: the literal ${VAR} form must never pass."""
    target = make_target("${OPENAI_API_KEY}", FakeClient(status_code=401))
    assert target.health_check() is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra security --extra test pytest tests/test_health_check.py -v`
Expected: `test_unreachable_endpoint_is_unhealthy` FAILS (returns True via the optimistic fallback). Others may pass already.

- [ ] **Step 3: Write the implementation**

In `surogate_eval/targets/model.py`, replace everything from the comment
`# Special handling for OpenAI and Anthropic APIs (non-localhost)` down to and including
`return bool(self.api_key)` (currently lines 218-235) with:

```python
            # Remote APIs: probe rather than trust that a credential is
            # present. A missing or rejected key must read as unhealthy,
            # not as "assume it works" (E-RUN-2). The old code returned
            # bool(self.api_key), so an unresolved "${OPENAI_API_KEY}"
            # literal counted as a valid credential and every request
            # then 401'd with all judged metrics scoring 0.
            if not self.api_key:
                logger.error(f"No API key provided for {self.name}")
                return False

            for path in ("/v1/models", "/models"):
                try:
                    response = self.client.get(path, timeout=10)
                except Exception:
                    continue
                if response.status_code == 200:
                    logger.debug(f"{self.name}: {path} probe healthy")
                    return True
                if response.status_code in (401, 403):
                    logger.error(
                        f"{self.name}: credential rejected by {path} "
                        f"(HTTP {response.status_code})"
                    )
                    return False

            logger.error(
                f"Could not verify {self.name} at {self.base_url}; "
                "treating as unhealthy"
            )
            return False
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --extra security --extra test pytest tests/test_health_check.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add surogate_eval/targets/model.py tests/test_health_check.py
git commit -m "fix(eval): treat unverifiable targets as unhealthy instead of assuming"
```

---

### Task 5: The DeepEval wrapper raises instead of fabricating a schema

**Files:**
- Modify: `surogate_eval/models/deepeval_wrapper.py:19-47` (delete `_empty_schema`), `:109-119`, `:178-180`
- Create: `tests/test_deepeval_wrapper.py`

**Interfaces:**
- Consumes: `JudgeParseError`, `JudgeUnavailableError` from Task 1.
- Produces: `DeepEvalTargetWrapper.generate()` raises `JudgeUnavailableError` or `JudgeParseError` instead of returning a fabricated schema. `_empty_schema` no longer exists.

- [ ] **Step 1: Write the failing test**

Create `tests/test_deepeval_wrapper.py`:

```python
import pytest
from pydantic import BaseModel

from surogate_eval.errors import JudgeParseError, JudgeUnavailableError
from surogate_eval.models.deepeval_wrapper import DeepEvalTargetWrapper
from surogate_eval.targets.base import TargetResponse


class Verdict(BaseModel):
    score: int


class FakeTarget:
    """A target under our control. Never touches the network."""

    def __init__(self, content="", error=None):
        self.name = "fake-judge"
        self.config = {"base_url": "https://api.openai.com/v1"}
        self._content = content
        self._error = error

    def send_request(self, request):
        return TargetResponse(
            content=self._content, raw_response={}, error=self._error,
        )


def test_target_error_raises_unavailable():
    wrapper = DeepEvalTargetWrapper(FakeTarget(error="HTTP 500"))
    with pytest.raises(JudgeUnavailableError):
        wrapper.generate("grade this", Verdict)


def test_empty_content_raises_unavailable():
    wrapper = DeepEvalTargetWrapper(FakeTarget(content=""))
    with pytest.raises(JudgeUnavailableError):
        wrapper.generate("grade this", Verdict)


def test_unparseable_content_raises_parse_error():
    """The judge answered in prose. Common with small judges."""
    wrapper = DeepEvalTargetWrapper(
        FakeTarget(content="I think the answer is pretty good overall.")
    )
    with pytest.raises(JudgeParseError):
        wrapper.generate("grade this", Verdict)


def test_valid_json_still_parses():
    wrapper = DeepEvalTargetWrapper(FakeTarget(content='{"score": 7}'))
    assert wrapper.generate("grade this", Verdict).score == 7


def test_markdown_wrapped_json_still_parses():
    wrapper = DeepEvalTargetWrapper(
        FakeTarget(content='```json\n{"score": 3}\n```')
    )
    assert wrapper.generate("grade this", Verdict).score == 3


def test_schemaless_call_raises_on_error():
    """Without a schema the old code returned "". Still an error."""
    wrapper = DeepEvalTargetWrapper(FakeTarget(error="HTTP 500"))
    with pytest.raises(JudgeUnavailableError):
        wrapper.generate("just text")


def test_empty_schema_helper_is_gone():
    """It fabricated malformed objects; nothing should resurrect it."""
    assert not hasattr(DeepEvalTargetWrapper, "_empty_schema")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra security --extra test pytest tests/test_deepeval_wrapper.py -v`
Expected: the four error tests FAIL (no exception raised), `test_empty_schema_helper_is_gone` FAILS.

- [ ] **Step 3: Write the implementation**

In `surogate_eval/models/deepeval_wrapper.py`, add the import:

```python
from ..errors import JudgeParseError, JudgeUnavailableError
```

Delete the entire `_empty_schema` staticmethod (currently lines 19-47). It fabricated objects that
were either a silent 0 or, for GEval, a malformed `Steps` instance that crashed downstream.

Replace the error and empty-content blocks (currently lines 109-119) with:

```python
        if response.error:
            raise JudgeUnavailableError(
                f"target {self.target.name!r} returned an error: {response.error}"
            )

        if not response.content:
            raise JudgeUnavailableError(
                f"target {self.target.name!r} returned empty content"
            )
```

Replace the final parse-failure fallback (currently lines 178-180) with:

```python
                    # All parsing strategies failed. Raise rather than
                    # inventing a score: the adapter turns this into an
                    # errored result (E-RUN-1).
                    raise JudgeParseError(
                        f"could not parse judge response from "
                        f"{self.target.name!r} into {schema.__name__}: "
                        f"{str(ex)[:200]}"
                    ) from ex
```

Note the enclosing `except (json.JSONDecodeError, Exception) as ex:` block: the raise must be the
last statement in it, after the truncated-JSON repair attempt.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --extra security --extra test pytest tests/test_deepeval_wrapper.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add surogate_eval/models/deepeval_wrapper.py tests/test_deepeval_wrapper.py
git commit -m "fix(eval): raise typed judge errors instead of fabricating empty schemas"
```

---

### Task 6: The adapter reports errors instead of scoring zero

**Files:**
- Modify: `surogate_eval/metrics/adapters/deepeval_adapter.py:198-207`, `:365-375`
- Create: `tests/test_deepeval_adapter.py`

**Interfaces:**
- Consumes: `JudgeError` (Task 1), `MetricResult.errored` (Task 2).
- Produces: `DeepEvalAdapter.evaluate()` returns an errored `MetricResult` on judge failure, on internal failure, and when the target's own request failed.

This is the load-bearing site. The spike showed exceptions propagate cleanly out of DeepEval's
`measure()`, and that this blanket `except Exception` is what flattened every failure to 0.0.

- [ ] **Step 1: Write the failing test**

Create `tests/test_deepeval_adapter.py`:

```python
import pytest

from surogate_eval.errors import JudgeUnavailableError
from surogate_eval.metrics.base import MetricStatus, MetricType
from surogate_eval.metrics.adapters.deepeval_adapter import DeepEvalAdapter
from surogate_eval.targets.base import TargetResponse


class Boom:
    """A deepeval metric stand-in that fails the way a broken judge does."""

    def __init__(self, exc):
        self.exc = exc

    def measure(self, test_case, _show_indicator=False):
        raise self.exc


def make_adapter(deepeval_metric):
    adapter = DeepEvalAdapter.__new__(DeepEvalAdapter)
    adapter.name = "correctness"
    adapter.metric_type = MetricType.G_EVAL
    adapter.config = {"deepeval_metric_type": "g_eval"}
    adapter.deepeval_metric = deepeval_metric
    adapter._judge_target = None
    adapter.is_conversational = False
    adapter.is_multimodal = False
    return adapter


def test_judge_error_is_errored_not_zero():
    adapter = make_adapter(Boom(JudgeUnavailableError("judge 500")))
    result = adapter.evaluate(object(), "some model output")
    assert result.status is MetricStatus.errored
    assert result.score is None


def test_internal_error_is_errored_and_labelled():
    """A bug in our code must not read as 'the model scored zero'."""
    adapter = make_adapter(Boom(AttributeError("'Steps' object has no attribute 'steps'")))
    result = adapter.evaluate(object(), "some model output")
    assert result.status is MetricStatus.errored
    assert result.metadata.get("error_kind") == "internal"


def test_failed_target_request_is_errored():
    adapter = make_adapter(Boom(RuntimeError("unreachable")))
    response = TargetResponse(content="", raw_response={}, error="HTTP 502")
    result = adapter.evaluate(object(), "", target_response=response)
    assert result.status is MetricStatus.errored
    assert "502" in result.reason


def test_genuinely_empty_completion_is_still_a_zero():
    """An empty answer with no transport error is a real bad answer."""
    adapter = make_adapter(Boom(RuntimeError("should not be reached")))
    response = TargetResponse(content="", raw_response={}, error=None)
    result = adapter.evaluate(object(), "", target_response=response)
    assert result.status is MetricStatus.scored
    assert result.score == 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra security --extra test pytest tests/test_deepeval_adapter.py -v`
Expected: first three FAIL (`status` is `scored`, `score` is 0.0).

- [ ] **Step 3: Write the implementation**

In `surogate_eval/metrics/adapters/deepeval_adapter.py`, add the import:

```python
from ...errors import JudgeError
```

`MetricResult` is already imported and is all the implementation needs. `MetricStatus` is used
only by the tests.

Replace the empty-output guard (currently lines 200-207) with:

```python
            # Check if we have actual output
            if not actual_output:
                # A failed request is a failure to measure, not a zero. An
                # empty completion with no transport error is a real (bad)
                # answer and stays a scored 0.0.
                if target_response is not None and target_response.error:
                    return MetricResult.errored(
                        metric_name=self.name,
                        metric_type=self.metric_type,
                        reason=f"Target request failed: {target_response.error}",
                        metadata={'error_kind': 'target'},
                    )
                return MetricResult(
                    metric_name=self.name,
                    metric_type=self.metric_type,
                    score=0.0,
                    success=False,
                    reason="No actual output to evaluate"
                )
```

Replace the blanket `except Exception` block (currently lines 365-375) with:

```python
        except JudgeError as e:
            # The judge broke. Reporting 0.0 here is what made a judge
            # outage indistinguishable from a bad model (E-RUN-1).
            logger.error(f"Judge failure in metric '{self.name}': {e}")
            return MetricResult.errored(
                metric_name=self.name,
                metric_type=self.metric_type,
                reason=f"Judge unavailable: {e}",
                metadata={
                    'deepeval_type': self.config.get('deepeval_metric_type'),
                    'error_kind': type(e).__name__,
                },
            )

        except Exception as e:
            # Our own bug. Still errored, but labelled so it is not
            # mistaken for a judge problem when reading results.
            logger.error(f"DeepEval evaluation failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return MetricResult.errored(
                metric_name=self.name,
                metric_type=self.metric_type,
                reason=f"Internal evaluation error: {e}",
                metadata={
                    'deepeval_type': self.config.get('deepeval_metric_type'),
                    'error_kind': 'internal',
                },
            )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --extra security --extra test pytest tests/test_deepeval_adapter.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add surogate_eval/metrics/adapters/deepeval_adapter.py tests/test_deepeval_adapter.py
git commit -m "fix(eval): report judge and internal failures as errored, not as a zero score"
```

---

### Task 7: Safety metrics fail closed

**Files:**
- Modify: `surogate_eval/metrics/safety.py` (three metrics, delete three helpers)
- Create: `tests/test_safety_metrics.py`

**Interfaces:**
- Consumes: `JudgeParseError` (Task 1), `MetricResult.errored` (Task 2).
- Produces: `ToxicityMetric`, `BiasMetric` and `HarmMetric` return errored results on judge failure. `_simple_toxicity_check`, `_simple_bias_check` and `_simple_harm_check` no longer exist.

- [ ] **Step 1: Write the failing test**

Create `tests/test_safety_metrics.py`:

```python
import pytest

from surogate_eval.metrics.base import MetricStatus, MetricType
from surogate_eval.metrics.safety import BiasMetric, HarmMetric, ToxicityMetric
from surogate_eval.targets.base import TargetResponse

PROSE = "Honestly this text seems fine to me, nothing concerning here."


class FakeJudge:
    def __init__(self, content=PROSE, error=None):
        self.name = "fake-judge"
        self._content = content
        self._error = error

    def send_request(self, request):
        return TargetResponse(
            content=self._content, raw_response={}, error=self._error,
        )


def build(cls, metric_type, judge):
    metric = cls({'name': 'm', 'type': metric_type.value, 'threshold': 0.5})
    metric.judge_target = judge
    return metric


TOXIC = "I will find you and hurt you, you worthless piece of trash."


@pytest.mark.parametrize(
    "cls,metric_type",
    [
        (ToxicityMetric, MetricType.TOXICITY),
        (BiasMetric, MetricType.BIAS),
        (HarmMetric, MetricType.HARM),
    ],
)
def test_unparseable_judge_output_is_errored(cls, metric_type):
    """The headline regression: a weak judge must not become a free pass."""
    metric = build(cls, metric_type, FakeJudge(content=PROSE))
    result = metric.evaluate(object(), TOXIC)
    assert result.status is MetricStatus.errored
    assert result.success is False
    assert result.score is None


@pytest.mark.parametrize(
    "cls,metric_type",
    [
        (ToxicityMetric, MetricType.TOXICITY),
        (BiasMetric, MetricType.BIAS),
        (HarmMetric, MetricType.HARM),
    ],
)
def test_missing_judge_is_errored(cls, metric_type):
    metric = build(cls, metric_type, None)
    metric.judge_target = None
    result = metric.evaluate(object(), TOXIC)
    assert result.status is MetricStatus.errored


@pytest.mark.parametrize(
    "cls,metric_type",
    [
        (ToxicityMetric, MetricType.TOXICITY),
        (BiasMetric, MetricType.BIAS),
        (HarmMetric, MetricType.HARM),
    ],
)
def test_judge_transport_error_is_errored(cls, metric_type):
    metric = build(cls, metric_type, FakeJudge(error="HTTP 500"))
    result = metric.evaluate(object(), TOXIC)
    assert result.status is MetricStatus.errored


def test_keyword_heuristics_are_gone():
    """They were five keywords, sixteen keywords, and an unconditional pass."""
    assert not hasattr(ToxicityMetric, "_simple_toxicity_check")
    assert not hasattr(BiasMetric, "_simple_bias_check")
    assert not hasattr(HarmMetric, "_simple_harm_check")


def test_valid_judge_output_still_scores():
    judge = FakeJudge(content='{"toxicity_score": 2, "reason": "mild"}')
    metric = build(ToxicityMetric, MetricType.TOXICITY, judge)
    result = metric.evaluate(object(), "some text")
    assert result.status is MetricStatus.scored
    assert result.score == pytest.approx(0.8)


def test_failed_target_request_is_errored():
    """Same rule as the adapter: a failed request is not a zero."""
    metric = build(ToxicityMetric, MetricType.TOXICITY, FakeJudge())
    response = TargetResponse(content="", raw_response={}, error="HTTP 502")
    result = metric.evaluate(object(), "", target_response=response)
    assert result.status is MetricStatus.errored


def test_genuinely_empty_output_is_still_a_zero():
    metric = build(ToxicityMetric, MetricType.TOXICITY, FakeJudge())
    response = TargetResponse(content="", raw_response={}, error=None)
    result = metric.evaluate(object(), "", target_response=response)
    assert result.status is MetricStatus.scored
    assert result.score == 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra security --extra test pytest tests/test_safety_metrics.py -v`
Expected: FAIL. In particular the bias cases return `success=True` today.

- [ ] **Step 3: Write the implementation**

In `surogate_eval/metrics/safety.py`, add the import:

```python
from ..errors import JudgeParseError
```

Delete these three methods entirely: `_simple_toxicity_check`, `_simple_bias_check`,
`_simple_harm_check`.

**All three metrics** open `evaluate` with the same empty-output guard, which returns a
`score=0.0`. Apply the same rule the adapter uses in Task 6. In each of `ToxicityMetric`,
`BiasMetric` and `HarmMetric`, replace:

```python
        if not actual_output:
            return MetricResult(
                metric_name=self.name,
                metric_type=self.metric_type,
                score=0.0,
                success=False,
                reason="No output to evaluate"
            )
```

with:

```python
        if not actual_output:
            # A failed request is a failure to measure. An empty completion
            # with no transport error is a real (bad) answer.
            if target_response is not None and target_response.error:
                return MetricResult.errored(
                    metric_name=self.name,
                    metric_type=self.metric_type,
                    reason=f"Target request failed: {target_response.error}",
                    metadata={'error_kind': 'target'},
                )
            return MetricResult(
                metric_name=self.name,
                metric_type=self.metric_type,
                score=0.0,
                success=False,
                reason="No output to evaluate"
            )
```

**ToxicityMetric.** Replace the no-judge branch:

```python
            if not self.judge_target:
                logger.warning("No judge target set, using simple heuristic")
                return self._simple_toxicity_check(actual_output)
```

with:

```python
            if not self.judge_target:
                return MetricResult.errored(
                    metric_name=self.name,
                    metric_type=self.metric_type,
                    reason="No judge target set; cannot assess toxicity.",
                    metadata={'error_kind': 'no_judge'},
                )
```

Replace the parse-failure branch:

```python
            except Exception as e:
                logger.warning(f"Failed to parse toxicity response: {e}")
                logger.debug(f"Raw response: {response.content[:300]}")
                return self._simple_toxicity_check(actual_output)
```

with:

```python
            except Exception as e:
                logger.warning(f"Failed to parse toxicity response: {e}")
                logger.debug(f"Raw response: {response.content[:300]}")
                raise JudgeParseError(
                    f"judge returned unparseable toxicity output: {e}"
                ) from e
```

Replace the outer handler:

```python
        except Exception as e:
            logger.error(f"Toxicity evaluation failed: {e}")
            return MetricResult(
                metric_name=self.name,
                metric_type=self.metric_type,
                score=0.0,
                success=False,
                reason=f"Evaluation error: {str(e)}"
            )
```

with:

```python
        except Exception as e:
            logger.error(f"Toxicity evaluation failed: {e}")
            return MetricResult.errored(
                metric_name=self.name,
                metric_type=self.metric_type,
                reason=f"Evaluation error: {e}",
                metadata={'error_kind': type(e).__name__},
            )
```

Also add the judge-transport guard immediately after `response = self.judge_target.send_request(request)`:

```python
            if response.error:
                raise JudgeParseError(f"judge request failed: {response.error}")
```

**BiasMetric.** Apply the same four edits. Replace the no-judge branch:

```python
                return self._simple_bias_check(actual_output)
```

with:

```python
                return MetricResult.errored(
                    metric_name=self.name,
                    metric_type=self.metric_type,
                    reason="No judge target set; cannot assess bias.",
                    metadata={'error_kind': 'no_judge'},
                )
```

Replace the parse-failure `return self._simple_bias_check(actual_output)` with:

```python
                raise JudgeParseError(
                    f"judge returned unparseable bias output: {e}"
                ) from e
```

Replace its outer handler's `MetricResult(... score=0.0 ...)` with:

```python
            return MetricResult.errored(
                metric_name=self.name,
                metric_type=self.metric_type,
                reason=f"Evaluation error: {e}",
                metadata={'error_kind': type(e).__name__},
            )
```

Add after its `send_request` call:

```python
            if response.error:
                raise JudgeParseError(f"judge request failed: {response.error}")
```

**HarmMetric.** Apply the same four edits. Replace the no-judge branch's
`return self._simple_harm_check(actual_output)` with:

```python
                return MetricResult.errored(
                    metric_name=self.name,
                    metric_type=self.metric_type,
                    reason="No judge target set; cannot assess harm.",
                    metadata={'error_kind': 'no_judge'},
                )
```

Replace the parse-failure `return self._simple_harm_check(actual_output)` with:

```python
                raise JudgeParseError(
                    f"judge returned unparseable harm output: {e}"
                ) from e
```

Replace its outer handler's `MetricResult(... score=0.0 ...)` with:

```python
            return MetricResult.errored(
                metric_name=self.name,
                metric_type=self.metric_type,
                reason=f"Evaluation error: {e}",
                metadata={'error_kind': type(e).__name__},
            )
```

Add after its `send_request` call:

```python
            if response.error:
                raise JudgeParseError(f"judge request failed: {response.error}")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --extra security --extra test pytest tests/test_safety_metrics.py -v`
Expected: PASS, 13 tests.

**Known remaining gap, do not fix here.** `custom_eval_backend.py:562-580` wraps its
`metric.measure(...)` in a broad `except Exception` that logs "G-Eval failed for row" and drops
the row. Task 5's wrapper now raises into that handler, so a judge failure there still silently
loses a row rather than recording it as errored. This is not a regression (the old fabricated
schema crashed into the same handler), and the custom-eval scoring path belongs to PR 12. Note it
in the PR description.

- [ ] **Step 5: Commit**

```bash
git add surogate_eval/metrics/safety.py tests/test_safety_metrics.py
git commit -m "fix(eval): fail safety metrics closed instead of falling back to keywords"
```

---

### Task 8: Run-level outcome and exit code

**Files:**
- Create: `surogate_eval/outcome.py`
- Modify: `surogate_eval/config/eval_config.py:269-275` (accept `max_error_rate`)
- Modify: `surogate_eval/eval.py:45-64` (`run` returns an exit code)
- Modify: `surogate_eval/cli/eval.py:104-107` (exit with it)
- Create: `tests/test_outcome.py`

**Interfaces:**
- Consumes: the `scored_n` / `errored_n` / `status` keys emitted by Task 2's `to_dict` methods.
- Produces: `compute_outcome(consolidated, max_error_rate) -> dict`, `exit_code_for(outcome) -> int`, `DEFAULT_MAX_ERROR_RATE = 0.2`, and `SurogateEval.run() -> int`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_outcome.py`:

```python
from surogate_eval.outcome import (
    DEFAULT_MAX_ERROR_RATE,
    compute_outcome,
    exit_code_for,
)


def batch(scored_n, errored_n):
    """Shape emitted by BatchMetricResult.to_dict()."""
    return {"scored_n": scored_n, "errored_n": errored_n, "results": []}


def target(status="success", **extra):
    return {"name": "t1", "status": status, **extra}


def test_default_threshold_is_twenty_percent():
    assert DEFAULT_MAX_ERROR_RATE == 0.2


def test_clean_run_completes():
    consolidated = {"targets": [target(evaluations=[batch(10, 0)])]}
    outcome = compute_outcome(consolidated)
    assert outcome["status"] == "completed"
    assert exit_code_for(outcome) == 0


def test_no_healthy_target_fails_regardless_of_threshold():
    consolidated = {"targets": [target(status="unhealthy")]}
    outcome = compute_outcome(consolidated)
    assert outcome["status"] == "failed"
    assert exit_code_for(outcome) == 1


def test_empty_targets_fails():
    outcome = compute_outcome({"targets": []})
    assert outcome["status"] == "failed"


def test_error_rate_over_threshold_fails():
    consolidated = {"targets": [target(evaluations=[batch(5, 5)])]}
    outcome = compute_outcome(consolidated)
    assert outcome["error_rate"] == 0.5
    assert outcome["status"] == "failed"
    assert exit_code_for(outcome) == 1


def test_error_rate_under_threshold_completes():
    consolidated = {"targets": [target(evaluations=[batch(90, 10)])]}
    outcome = compute_outcome(consolidated)
    assert outcome["error_rate"] == 0.1
    assert outcome["status"] == "completed"


def test_threshold_is_configurable():
    consolidated = {"targets": [target(evaluations=[batch(90, 10)])]}
    outcome = compute_outcome(consolidated, max_error_rate=0.05)
    assert outcome["status"] == "failed"


def test_counts_are_not_double_counted():
    """A batch dict carries both summary counts and a results list."""
    nested = {
        "scored_n": 1,
        "errored_n": 1,
        "results": [
            {"metric_name": "m", "status": "scored"},
            {"metric_name": "m", "status": "errored"},
        ],
    }
    outcome = compute_outcome({"targets": [target(evaluations=[nested])]})
    assert outcome["scored"] == 1
    assert outcome["errored"] == 1


def test_bare_metric_results_are_counted():
    """Paths that emit MetricResult dicts without a batch wrapper."""
    consolidated = {
        "targets": [target(evaluations=[
            {"metric_name": "m", "status": "errored"},
            {"metric_name": "m", "status": "scored"},
        ])]
    }
    outcome = compute_outcome(consolidated)
    assert outcome["scored"] == 1
    assert outcome["errored"] == 1


def test_deeply_nested_counts_are_found():
    consolidated = {
        "targets": [target(benchmarks=[{"suite": {"metrics": [batch(3, 1)]}}])]
    }
    outcome = compute_outcome(consolidated)
    assert outcome["scored"] == 3
    assert outcome["errored"] == 1


def test_failure_reason_is_populated():
    consolidated = {"targets": [target(evaluations=[batch(0, 10)])]}
    outcome = compute_outcome(consolidated)
    assert outcome["reason"]
    assert "10" in outcome["reason"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra security --extra test pytest tests/test_outcome.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'surogate_eval.outcome'`

- [ ] **Step 3: Write the implementation**

Create `surogate_eval/outcome.py`:

```python
"""Run-level outcome: did this evaluation produce trustworthy results?

The runner used to end every run with "completed" and exit 0, even when no
target was reachable and every metric had errored. Ops reads the exit code
to decide whether a run failed, so the outcome computed here is what makes
a broken run visible.
"""

from typing import Any, Dict, Tuple

DEFAULT_MAX_ERROR_RATE = 0.2


def _collect_counts(node: Any) -> Tuple[int, int]:
    """Sum (scored, errored) across an arbitrarily nested results tree.

    A ``BatchMetricResult`` dict carries both summary counts and the
    individual results that produced them, so when the summary keys are
    present we take those and do NOT descend, or every case is counted
    twice.
    """
    if isinstance(node, dict):
        if 'scored_n' in node and 'errored_n' in node:
            return int(node['scored_n']), int(node['errored_n'])
        if 'metric_name' in node and 'status' in node:
            return (1, 0) if node['status'] == 'scored' else (0, 1)
        scored = errored = 0
        for value in node.values():
            s, e = _collect_counts(value)
            scored += s
            errored += e
        return scored, errored

    if isinstance(node, list):
        scored = errored = 0
        for item in node:
            s, e = _collect_counts(item)
            scored += s
            errored += e
        return scored, errored

    return 0, 0


def compute_outcome(
        consolidated: Dict[str, Any],
        max_error_rate: float = DEFAULT_MAX_ERROR_RATE,
) -> Dict[str, Any]:
    """Decide whether a finished run should be reported as failed."""
    targets = consolidated.get('targets') or []
    healthy = [t for t in targets if t.get('status') == 'success']

    scored, errored = _collect_counts(targets)
    total = scored + errored
    error_rate = (errored / total) if total else 0.0

    status = 'completed'
    reason = None

    if not healthy:
        status = 'failed'
        reason = 'No target completed its evaluations.'
    elif error_rate > max_error_rate:
        status = 'failed'
        reason = (
            f'Error rate {error_rate:.1%} exceeds the maximum '
            f'{max_error_rate:.1%} ({errored} of {total} evaluations errored).'
        )

    return {
        'status': status,
        'reason': reason,
        'scored': scored,
        'errored': errored,
        'error_rate': round(error_rate, 4),
        'max_error_rate': max_error_rate,
    }


def exit_code_for(outcome: Dict[str, Any]) -> int:
    """0 when the run is trustworthy, 1 otherwise."""
    return 0 if outcome.get('status') == 'completed' else 1
```

In `surogate_eval/config/eval_config.py`, the `EvalConfig` class reads only `project` and
`targets`, so a top-level `max_error_rate` would be silently dropped. Add the field declaration
beside the existing ones:

```python
    project: Optional[ProjectConfig] = None
    targets: Optional[List[TargetConfig]] = None
    max_error_rate: Optional[float] = None
```

and read it in `__init__`, before `self.__post_init__()`:

```python
        self.max_error_rate = cfg['max_error_rate']
```

In `surogate_eval/eval.py`, add the import:

```python
from surogate_eval.outcome import DEFAULT_MAX_ERROR_RATE, compute_outcome, exit_code_for
```

Replace `run` (currently lines 45-64) with:

```python
    def run(self) -> int:
        """Run the evaluation pipeline. Returns a process exit code."""
        from datetime import datetime

        logger.banner("SUROGATE EVAL")

        self.consolidated_results["timestamp"] = datetime.now().isoformat()
        self.consolidated_results["project"] = {
            "name": self.config.project.name,
            "version": self.config.project.version,
            "description": self.config.project.description,
        }

        try:
            self._process_targets()
        finally:
            self._cleanup()

        configured = self.config.max_error_rate
        outcome = compute_outcome(
            self.consolidated_results,
            DEFAULT_MAX_ERROR_RATE if configured is None else float(configured),
        )
        self.consolidated_results["outcome"] = outcome

        self._save_consolidated_results()

        if outcome["status"] == "completed":
            logger.success("Surogate Eval completed")
        else:
            logger.error(f"Surogate Eval failed: {outcome['reason']}")

        return exit_code_for(outcome)
```

In `surogate_eval/cli/eval.py`, replace the final call (currently lines 104-107):

```python
        SurogateEval(
            config=config,
            args=command_args,
        ).run()
```

with:

```python
        sys.exit(
            SurogateEval(
                config=config,
                args=command_args,
            ).run()
        )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --extra security --extra test pytest tests/test_outcome.py -v`
Expected: PASS, 11 tests.

- [ ] **Step 5: Run the whole suite**

Run: `uv run --extra security --extra test pytest tests/ -v`
Expected: PASS, all tests from Tasks 1 to 8.

- [ ] **Step 6: Commit**

```bash
git add surogate_eval/outcome.py surogate_eval/eval.py surogate_eval/cli/eval.py \
        surogate_eval/config/eval_config.py tests/test_outcome.py
git commit -m "feat(eval): fail the run and exit non-zero when results are untrustworthy"
```

---

## After the plan

1. **Simplify pass.** Run the `simplify` skill over the diff before review.
2. **Code review.** `/code-review`.
3. **Live verification.** This is a hard gate, not optional. Run a real eval against a
   deliberately broken judge (bad `baseUrl`) and confirm: the run reports failed, the process
   exits non-zero, and ops marks the run failed via the PR #308 exit-code gate. Then run a
   healthy eval and confirm it still completes and ingests normally.
4. **Findings doc correction.** Update the E-RUN-1 entry in
   `Misc/Training-Dataset-Eval/eval-findings.md`: the load-bearing site is the adapter's blanket
   `except`, not the wrapper. Note that `_empty_schema` produced a malformed `Steps` object for
   GEval rather than a clean zero.
5. **Only then** open the PR.
