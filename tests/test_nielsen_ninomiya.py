import pytest

from validators.nielsen_ninomiya import (
    cannot_have_single_chirality,
    chiral_sum_zero_satisfiable,
)


@pytest.mark.parametrize("n", [2, 4, 6, 8])
def test_even_nodes_satisfiable(n):
    assert chiral_sum_zero_satisfiable(n)


@pytest.mark.parametrize("n", [3, 5, 7])
def test_odd_nodes_unsatisfiable(n):
    assert not chiral_sum_zero_satisfiable(n)


@pytest.mark.parametrize("n", [2, 4, 6])
def test_no_single_chirality(n):
    assert cannot_have_single_chirality(n)
