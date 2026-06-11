import pytest

from validators.graph_chiral_balance import (
    approximate_pairing_bound_example,
    bipartite_index_bound_holds,
    chiral_breaking_lifts_pairing,
    chiral_disorder_preserves_pairing,
    chiral_pairing_finite_matrix,
    generic_matching_nullity_holds,
    irregular_graph_locality,
    single_site_bulk_mode_requires_edge_cut,
    ssh_open_chain_zero_modes,
    structural_zero_modes_stay_in_epsilon_window,
)


@pytest.mark.parametrize("n_a,n_b,seed", [(3, 3, 0), (4, 3, 1), (2, 5, 2)])
def test_chiral_pairing_finite_matrix(n_a, n_b, seed):
    assert chiral_pairing_finite_matrix(n_a, n_b, seed)


@pytest.mark.parametrize("n_a,n_b,seed", [(2, 5, 3), (5, 2, 4), (4, 4, 5), (6, 3, 6)])
def test_bipartite_index_bound(n_a, n_b, seed):
    assert bipartite_index_bound_holds(n_a, n_b, seed)


@pytest.mark.parametrize(
    "n_a,n_b,edges",
    [
        (2, 1, [(0, 0), (1, 0)]),
        (3, 3, [(0, 0), (1, 0), (1, 1), (2, 2)]),
        (4, 3, [(0, 0), (1, 0), (1, 1), (2, 1), (3, 2)]),
    ],
)
def test_generic_matching_nullity(n_a, n_b, edges):
    assert generic_matching_nullity_holds(n_a, n_b, edges)


def test_ssh_open_chain_zero_modes():
    assert ssh_open_chain_zero_modes()


@pytest.mark.parametrize("seed", [0, 7, 11])
def test_chiral_disorder_preserves_pairing(seed):
    assert chiral_disorder_preserves_pairing(seed=seed)


def test_chiral_breaking_lifts_pairing():
    assert chiral_breaking_lifts_pairing()


def test_approximate_pairing_bound():
    assert approximate_pairing_bound_example()


def test_structural_zero_modes_stay_in_epsilon_window():
    assert structural_zero_modes_stay_in_epsilon_window()


def test_single_site_bulk_mode_requires_edge_cut():
    assert single_site_bulk_mode_requires_edge_cut()


def test_irregular_graph_locality():
    assert irregular_graph_locality()
