from examples.strategy_compare import DELTAS, HORIZON, INITIAL, SCENARIOS, report
from compound_curves import compare_scenarios, rank_sensitivities


def test_long_horizon_example_is_deterministic_and_complete():
    first = report()
    assert first == report()
    assert "scenario ranking:" in first
    assert "dominant parameters by scenario:" in first
    assert set(SCENARIOS) == {"steady", "high_growth"}
    assert HORIZON == 120


def test_example_ranks_scenarios_and_parameter_impacts():
    ranked = compare_scenarios(INITIAL, SCENARIOS)
    assert ranked[0].terminal >= ranked[1].terminal
    for name, steps in SCENARIOS.items():
        impacts = rank_sensitivities(INITIAL, steps, DELTAS)
        magnitudes = tuple(abs(item.terminal_delta) for item in impacts)
        assert magnitudes == tuple(sorted(magnitudes, reverse=True))
        assert {item.parameter for item in impacts} == set(DELTAS)
        assert magnitudes[0] > magnitudes[-1]
