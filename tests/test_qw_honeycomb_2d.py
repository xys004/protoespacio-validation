from validators.qw_honeycomb_2d import (
    anticommutators_give_inner_product,
    effective_dispersion_is_isotropic_cone,
    effective_hamiltonian_is_dirac_2d,
    projection_gives_dirac_x,
    projection_gives_dirac_y,
    sum_of_tau_is_zero,
    tau_matrices_hermitian,
)


def test_tau_hermitian():
    assert tau_matrices_hermitian()


def test_tau_sums_to_zero():
    assert sum_of_tau_is_zero()


def test_anticommutator_inner_product():
    assert anticommutators_give_inner_product()


def test_projection_x():
    assert projection_gives_dirac_x()


def test_projection_y():
    assert projection_gives_dirac_y()


def test_h_eff_is_dirac_2d():
    assert effective_hamiltonian_is_dirac_2d()


def test_h_eff_dispersion():
    assert effective_dispersion_is_isotropic_cone()
