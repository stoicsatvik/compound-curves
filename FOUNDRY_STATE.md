# Foundry State

## Objective
Build a reusable compounding simulator for arbitrary stocks such as capital, skill, audience, reputation, code reuse, and knowledge, with gains, decay, shocks, saturation, and reinvestment.

## Public boundary
Expose generic mathematics and examples only. Keep private any proprietary scoring, personal/private financial data, or strategy engines built on top.

## V0 milestone
- Discrete-time compounding model
- Contributions, withdrawals, decay, shocks, and capped/logistic growth
- Scenario comparison and sensitivity analysis
- CSV/JSON input and reproducible outputs
- Tests for invariants and edge cases

## Acceptance
A user can compare two long-horizon strategies and see which parameters dominate the result rather than merely plotting compound interest.

## Next move
Implement the core state-transition model and property tests before visualization.

Status: ACTIVE / NOT YET PROVEN
