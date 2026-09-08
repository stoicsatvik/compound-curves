from math import isclose

from compound_curves import Step, logistic_transition, sensitivity, simulate, transition


def test_zero_step_is_identity():
    assert transition(100.0, Step()) == 100.0


def test_compounding_is_deterministic():
    steps = (Step(contribution=10, rate=.10),) * 3
    assert simulate(100, steps) == simulate(100, steps)
    assert round(simulate(100, steps)[-1], 6) == 169.51


def test_withdrawal_decay_and_shock_order():
    # Decimal 0.1 is not exactly representable in binary floating point;
    # assert the mathematical contract without requiring an exact bit pattern.
    assert isclose(
        transition(100, Step(contribution=20, withdrawal=10, rate=.1, decay=.1, shock=-4)),
        104.9,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_floor_and_cap_are_invariants():
    assert transition(10, Step(withdrawal=100)) == 0
    assert transition(100, Step(rate=1), cap=150) == 150


def test_logistic_growth_respects_capacity():
    value = 90
    for _ in range(100):
        value = logistic_transition(value, rate=.5, capacity=100)
    assert 99.99 < value <= 100


def test_sensitivity_reports_terminal_delta():
    steps = (Step(rate=.05),) * 10
    assert sensitivity(100, steps, "rate", .01) > 0


def test_invalid_parameters_fail_closed():
    for step in (Step(decay=1.1), Step(rate=-1.1)):
        try:
            transition(100, step)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid transition accepted")
