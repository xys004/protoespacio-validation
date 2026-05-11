import pytest

from validators.protoespacio_minimo import (
    chain_extension_exists,
    cyclic_neighborhood_blocks_total_order,
    linear_chain_is_partial_order,
    two_cycle_breaks_antisymmetry,
)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_linear_chain_partial_order(n):
    assert linear_chain_is_partial_order(n)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_two_cycle_unsat(n):
    assert two_cycle_breaks_antisymmetry(n)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_chain_extension(n):
    assert chain_extension_exists(n)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_cyclic_unsat(n):
    assert cyclic_neighborhood_blocks_total_order(n)