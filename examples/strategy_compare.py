"""Deterministic long-horizon comparison showing parameter dominance."""
from compound_curves import Step, compare_scenarios, rank_sensitivities

INITIAL = 100.0
HORIZON = 120
SCENARIOS = {
    "steady": (Step(contribution=8.0, rate=0.025, decay=0.005),) * HORIZON,
    "high_growth": (Step(contribution=3.0, rate=0.04, decay=0.012),) * HORIZON,
}
DELTAS = {"contribution": 1.0, "rate": 0.005, "decay": 0.005}


def report() -> str:
    ranked = compare_scenarios(INITIAL, SCENARIOS)
    lines = ["scenario ranking:"]
    lines.extend(f"{i}. {r.name}: terminal={r.terminal:.6f}" for i, r in enumerate(ranked, 1))
    lines.append("dominant parameters by scenario:")
    for name in sorted(SCENARIOS):
        impacts = rank_sensitivities(INITIAL, SCENARIOS[name], DELTAS)
        rendered = ", ".join(f"{x.parameter}={x.terminal_delta:+.6f}" for x in impacts)
        lines.append(f"{name}: {rendered}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
