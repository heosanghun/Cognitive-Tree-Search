from cts.types import NuVector, RuntimeBudgetState


def test_budget_clone_independent():
    b = RuntimeBudgetState(ado_accumulated=1.0)
    c = b.clone()
    c.ado_accumulated = 2.0
    assert b.ado_accumulated == 1.0
