import pytest

from validators.rama_estructural import (
    chirality_anticommutes_with_bipartite_adjacency,
    disordered_ssh_preserves_chirality,
    disordered_ssh_spectrum_remains_paired,
    index_holds_on_subgraph_of_K_n_m,
    index_theorem_lower_bound_K_n_m,
    irregular_graph_locality,
    matching_graph_index_is_exact,
    ssh_finite_spectrum_is_chiral_symmetric,
    tight_binding_chain_first_order_is_local,
)


# E1: localidad en grafos no periodicos

@pytest.mark.parametrize("n", [4, 5, 6])
def test_chain_locality(n):
    assert tight_binding_chain_first_order_is_local(n)


def test_irregular_graph_locality():
    assert irregular_graph_locality()


# E2: quiralidad bajo perturbacion local

def test_ssh_chiral_symmetric():
    assert ssh_finite_spectrum_is_chiral_symmetric()


def test_disordered_ssh_chirality_preserved():
    assert disordered_ssh_preserves_chirality()


def test_disordered_ssh_spectrum_paired():
    assert disordered_ssh_spectrum_remains_paired()


# E3: indice quiral en bipartito sin toro

@pytest.mark.parametrize("n,m", [(2, 3), (3, 3), (4, 2), (5, 3)])
def test_index_lower_bound_complete_bipartite(n, m):
    assert index_theorem_lower_bound_K_n_m(n, m)


@pytest.mark.parametrize("n,m", [(2, 3), (3, 3), (4, 2), (5, 3), (1, 5)])
def test_matching_graph_exact_index(n, m):
    assert matching_graph_index_is_exact(n, m)


@pytest.mark.parametrize("n,m", [(2, 3), (3, 3), (4, 2)])
def test_chirality_anticommutes(n, m):
    assert chirality_anticommutes_with_bipartite_adjacency(n, m)


def test_index_on_subgraph():
    assert index_holds_on_subgraph_of_K_n_m()
