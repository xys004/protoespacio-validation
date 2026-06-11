from validators.induced_gravity import (
    cosmological_term_is_dimensionless,
    dirac_a1_density_is_minus_R_over_12,
    einstein_hilbert_action_is_dimensionless,
    einstein_hilbert_term_is_generated,
    higher_curvature_terms_are_cutoff_independent,
    inverse_newton_has_mass_squared_dimension,
    inverse_newton_is_linear_in_fermion_count,
    inverse_newton_scales_as_cutoff_squared,
)


def test_dirac_a1():
    assert dirac_a1_density_is_minus_R_over_12()


def test_eh_generated():
    assert einstein_hilbert_term_is_generated()


def test_inverse_newton_linear_in_N():
    assert inverse_newton_is_linear_in_fermion_count()


def test_inverse_newton_cutoff_squared():
    assert inverse_newton_scales_as_cutoff_squared()


def test_eh_action_dimensionless():
    assert einstein_hilbert_action_is_dimensionless()


def test_inverse_newton_mass_squared():
    assert inverse_newton_has_mass_squared_dimension()


def test_cosmological_dimensionless():
    assert cosmological_term_is_dimensionless()


def test_higher_curvature_cutoff_independent():
    assert higher_curvature_terms_are_cutoff_independent()
