import pytest

from validators.nielsen_ninomiya import (
    balanced_count_required,
    cannot_have_uniform_chirality,
    chiral_sum_zero_satisfiable,
)


@pytest.mark.parametrize("n", [2, 4, 6, 8])
def test_even_nodes_satisfiable(n):
    assert chiral_sum_zero_satisfiable(n)


@pytest.mark.parametrize("n", [3, 5, 7])
def test_odd_nodes_unsatisfiable(n):
    assert not chiral_sum_zero_satisfiable(n)


@pytest.mark.parametrize("n", [2, 4, 6])
def test_no_uniform_chirality(n):
    assert cannot_have_uniform_chirality(n)


@pytest.mark.parametrize("n", [2, 4, 6, 8])
def test_balanced_count(n):
    assert balanced_count_required(n)
