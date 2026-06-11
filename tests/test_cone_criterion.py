import pytest

from validators.cone_criterion import (
    cycle_cone_defect_is_zero,
    difference_operator_is_local,
    dirac_operator_hosts_internal_algebra,
    dirac_square_equals_laplacian,
    eigenvalue_stability_under_perturbation,
    laplacian_is_positive_semidefinite,
    scalar_hamiltonian_lacks_internal_algebra,
)


@pytest.mark.parametrize("n", [4, 5, 6, 8])
def test_dirac_square_is_laplacian(n):
    assert dirac_square_equals_laplacian(n)


@pytest.mark.parametrize("n", [4, 6, 8])
def test_difference_operator_local(n):
    assert difference_operator_is_local(n)


@pytest.mark.parametrize("n", [4, 6, 8])
def test_cone_defect_zero(n):
    assert cycle_cone_defect_is_zero(n)


@pytest.mark.parametrize("size", [4, 5, 6])
def test_eigenvalue_stability(size):
    assert eigenvalue_stability_under_perturbation(size)


@pytest.mark.parametrize("n", [3, 4, 6, 8])
def test_laplacian_psd(n):
    assert laplacian_is_positive_semidefinite(n)


def test_scalar_negative_control():
    assert scalar_hamiltonian_lacks_internal_algebra()


def test_dirac_hosts_internal_algebra():
    assert dirac_operator_hosts_internal_algebra()
