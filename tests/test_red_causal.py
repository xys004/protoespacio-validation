import pytest

from validators.red_causal import (
    ball_strictly_increases_with_n,
    cycle_covers_in_finite_steps,
    linear_chain_reachability,
    locality_implies_finite_propagation,
)


@pytest.mark.parametrize("n_steps,chain_len", [(2, 5), (3, 6), (1, 4)])
def test_linear_chain(n_steps, chain_len):
    assert linear_chain_reachability(n_steps, chain_len)


@pytest.mark.parametrize("n", [4, 5, 6, 8])
def test_cycle_covers(n):
    assert cycle_covers_in_finite_steps(n)


@pytest.mark.parametrize("n_steps", [1, 2, 3])
def test_locality_propagation(n_steps):
    assert locality_implies_finite_propagation(n_steps)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_ball_strictly_grows(n):
    assert ball_strictly_increases_with_n(n)
