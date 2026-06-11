import pytest

from validators.laplacian_projector import (
    constant_mode_is_always_low,
    cycle_laplacian,
    cycle_laplacian_eigenvalues_exact,
    cycle_low_mode_quadratic,
    graph_laplacian,
    projector_is_idempotent_hermitian,
    projector_rank_counts_low_modes,
)

# Complete graph K_4: rational eigenvectors (spectrum {0, 4, 4, 4}).
_K4_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def _k4():
    return graph_laplacian(4, _K4_EDGES)


@pytest.mark.parametrize("n", [3, 4, 6])
def test_cycle_spectrum_exact(n):
    assert cycle_laplacian_eigenvalues_exact(n)


def test_low_mode_quadratic():
    assert cycle_low_mode_quadratic()


# Projector properties on graphs with rational eigenspaces (C_4, K_4):
# these keep the exact sympy projector assembly fast.
@pytest.mark.parametrize("threshold", [0.5, 2.5])
def test_projector_idempotent_hermitian_cycle(threshold):
    assert projector_is_idempotent_hermitian(cycle_laplacian(4), threshold)


@pytest.mark.parametrize("threshold", [0.5, 2.5])
def test_projector_rank_cycle(threshold):
    assert projector_rank_counts_low_modes(cycle_laplacian(4), threshold)


@pytest.mark.parametrize("threshold", [0.5, 4.5])
def test_projector_idempotent_hermitian_complete(threshold):
    assert projector_is_idempotent_hermitian(_k4(), threshold)


@pytest.mark.parametrize("threshold", [0.5, 4.5])
def test_projector_rank_complete(threshold):
    assert projector_rank_counts_low_modes(_k4(), threshold)


def test_constant_mode_low():
    assert constant_mode_is_always_low()
