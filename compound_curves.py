"""Deterministic discrete-time compounding primitives."""
from dataclasses import dataclass
from math import exp, isfinite
from typing import Iterable

@dataclass(frozen=True)
class Step:
    contribution: float = 0.0
    withdrawal: float = 0.0
    rate: float = 0.0
    decay: float = 0.0
    shock: float = 0.0


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
