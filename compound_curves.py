"""Deterministic discrete-time compounding primitives."""
from dataclasses import asdict, dataclass
import csv
import io
import json
from math import isfinite
from typing import Iterable

@dataclass(frozen=True)
class Step:
    contribution: float = 0.0
    withdrawal: float = 0.0
    rate: float = 0.0
    decay: float = 0.0
    shock: float = 0.0


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    terminal: float
    history: tuple[float, ...]


@dataclass(frozen=True)
class SensitivityResult:
    parameter: str
    terminal_delta: float


@dataclass(frozen=True)
class NormalizedSensitivityResult:
    parameter: str
    normalized_impact: float
    terminal_delta: float


def transition(stock: float, step: Step, *, cap: float | None = None) -> float:
    """Apply flows, multiplicative gain/decay, shock, then optional hard cap."""
    values = (stock, step.contribution, step.withdrawal, step.rate, step.decay, step.shock)
    if not all(isfinite(v) for v in values):
        raise ValueError("all values must be finite")
    if stock < 0 or step.decay < 0 or step.decay > 1 or step.rate < -1:
        raise ValueError("invalid stock, decay, or rate")
    if cap is not None and (not isfinite(cap) or cap < 0):
        raise ValueError("cap must be finite and non-negative")
    value = (stock + step.contribution - step.withdrawal) * (1 + step.rate) * (1 - step.decay) + step.shock
    value = max(0.0, value)
    return min(value, cap) if cap is not None else value


def logistic_transition(stock: float, *, rate: float, capacity: float, contribution: float = 0.0) -> float:
    """One logistic-growth step with an additive contribution before growth."""
    if stock < 0 or rate < 0 or capacity <= 0 or not all(isfinite(v) for v in (stock, rate, capacity, contribution)):
        raise ValueError("invalid logistic parameters")
    base = max(0.0, stock + contribution)
    return min(capacity, max(0.0, base + rate * base * (1 - base / capacity)))


def simulate(initial: float, steps: Iterable[Step], *, cap: float | None = None) -> tuple[float, ...]:
    history = [initial]
    current = initial
    for step in steps:
        current = transition(current, step, cap=cap)
        history.append(current)
    return tuple(history)


def sensitivity(initial: float, steps: tuple[Step, ...], parameter: str, delta: float, *, cap: float | None = None) -> float:
    """Return terminal-value change when one Step field is shifted everywhere."""
    if parameter not in Step.__dataclass_fields__ or not isfinite(delta):
        raise ValueError("unknown parameter or invalid delta")
    baseline = simulate(initial, steps, cap=cap)[-1]
    changed = tuple(Step(**{**s.__dict__, parameter: getattr(s, parameter) + delta}) for s in steps)
    return simulate(initial, changed, cap=cap)[-1] - baseline


def compare_scenarios(initial: float, scenarios: dict[str, tuple[Step, ...]], *, cap: float | None = None) -> tuple[ScenarioResult, ...]:
    """Evaluate named scenarios and rank by terminal stock, then name for stable ties."""
    if not scenarios:
        raise ValueError("at least one scenario is required")
    results = []
    for name, steps in scenarios.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("scenario names must be non-empty strings")
        history = simulate(initial, steps, cap=cap)
        results.append(ScenarioResult(name=name, terminal=history[-1], history=history))
    return tuple(sorted(results, key=lambda result: (-result.terminal, result.name)))


def rank_sensitivities(initial: float, steps: tuple[Step, ...], deltas: dict[str, float], *, cap: float | None = None) -> tuple[SensitivityResult, ...]:
    """Rank explicit parameter perturbations by absolute terminal impact."""
    if not deltas:
        raise ValueError("at least one parameter delta is required")
    results = tuple(
        SensitivityResult(parameter=parameter, terminal_delta=sensitivity(initial, steps, parameter, delta, cap=cap))
        for parameter, delta in deltas.items()
    )
    return tuple(sorted(results, key=lambda result: (-abs(result.terminal_delta), result.parameter)))


def rank_normalized_sensitivities(initial: float, steps: tuple[Step, ...], scales: dict[str, float], *, fraction: float = 0.01, cap: float | None = None) -> tuple[NormalizedSensitivityResult, ...]:
    """Rank parameters using equal fractional perturbations of explicit reference scales.

    The score is terminal change divided by baseline terminal value and perturbation
    fraction. Supplying scales in the same units as each parameter makes the ranking
    invariant to a consistent change of units for that parameter.
    """
    if not scales or not isfinite(fraction) or fraction <= 0:
        raise ValueError("scales and a positive finite fraction are required")
    baseline = simulate(initial, steps, cap=cap)[-1]
    if baseline <= 0 or not isfinite(baseline):
        raise ValueError("positive finite baseline terminal value is required")
    results = []
    for parameter, scale in scales.items():
        if parameter not in Step.__dataclass_fields__ or not isfinite(scale) or scale <= 0:
            raise ValueError("unknown parameter or invalid reference scale")
        terminal_delta = sensitivity(initial, steps, parameter, fraction * scale, cap=cap)
        results.append(NormalizedSensitivityResult(parameter, terminal_delta / baseline / fraction, terminal_delta))
    return tuple(sorted(results, key=lambda result: (-abs(result.normalized_impact), result.parameter)))


def scenarios_to_json(initial: float, scenarios: dict[str, tuple[Step, ...]]) -> str:
    """Canonical JSON input representation, stable across mapping insertion order."""
    if not scenarios:
        raise ValueError("at least one scenario is required")
    if isinstance(initial, bool) or not isinstance(initial, (int, float)) or not isfinite(initial):
        raise ValueError("initial must be a finite JSON number")
    payload = {"initial": initial, "scenarios": {name: [asdict(step) for step in scenarios[name]] for name in sorted(scenarios)}}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _reject_json_constant(value: str) -> None:
    """Reject non-standard NaN/Infinity constants accepted by Python's decoder."""
    raise ValueError(f"non-finite JSON constant: {value}")


def scenarios_from_json(text: str) -> tuple[float, dict[str, tuple[Step, ...]]]:
    """Parse canonical scenario JSON without changing valid JSON number representation."""
    try:
        payload = json.loads(text, parse_constant=_reject_json_constant)
        initial = payload["initial"]
        if isinstance(initial, bool) or not isinstance(initial, (int, float)) or not isfinite(initial):
            raise ValueError("initial must be a finite JSON number")
        raw_scenarios = payload["scenarios"]
        if not isinstance(raw_scenarios, dict):
            raise ValueError("scenarios must be an object")
        scenarios = {name: tuple(Step(**step) for step in steps) for name, steps in raw_scenarios.items()}
        compare_scenarios(initial, scenarios)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid scenario JSON") from exc
    return initial, scenarios


def results_to_csv(results: tuple[ScenarioResult, ...]) -> str:
    """Export deterministic ranked results as newline-stable CSV."""
    if not results:
        raise ValueError("at least one result is required")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("rank", "name", "terminal", "history_json"))
    for rank, result in enumerate(results, 1):
        writer.writerow((rank, result.name, repr(result.terminal), json.dumps(result.history, separators=(",", ":"), allow_nan=False)))
    return stream.getvalue()
