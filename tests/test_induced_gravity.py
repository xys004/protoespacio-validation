from validators.induced_gravity import (
    a1_uses_operator_derived_sign,
    boson_statistics_flips_induced_sign,
    conformal_coupling_kills_EH_term,
    cosmological_term_is_dimensionless,
    derived_newton_scaling_matches_convenience_form,
    dirac_a1_density_is_minus_R_over_12,
    dirac_induced_eh_coefficient_is_negative_definite,
    einstein_hilbert_action_is_dimensionless,
    einstein_hilbert_term_is_generated,
    gaussian_scheme_same_power_and_sign_different_constant,
    higher_curvature_terms_are_cutoff_independent,
    inverse_newton_has_mass_squared_dimension,
    inverse_newton_is_linear_in_fermion_count,
    inverse_newton_scales_as_cutoff_squared,
    newton_sign_is_not_a_theorem_without_scheme_input,
    proper_time_a0_integral_is_lambda4_over_64pi2,
    proper_time_a1_integral_is_lambda2_over_32pi2,
    pure_dirac_content_is_sign_definite,
    wrong_sign_E_gives_five_twelfths,
)


def test_dirac_a1():
    """a_1(E = R/4) = -R/12 per spinor component."""
    assert dirac_a1_density_is_minus_R_over_12()


def test_a1_operator_derived_sign():
    """The a_1 = -R/12 chain consumes the SIGNED E from lichnerowicz.py."""
    assert a1_uses_operator_derived_sign()


def test_wrong_sign_five_twelfths():
    """a_1(E = -R/4) = +5R/12: sign conventions are distinguishable, not mirrors."""
    assert wrong_sign_E_gives_five_twelfths()


def test_conformal_coupling_kills_EH():
    """a_1(E = R/6) = 0: the conformal-coupling counterfactual erases the EH term."""
    assert conformal_coupling_kills_EH_term()


def test_eh_generated():
    """The a_1 coefficient of R is a pure non-zero number: the EH term is forced."""
    assert einstein_hilbert_term_is_generated()


def test_proper_time_a0_integral():
    """Lambda^4 cosmological scaling from an actual convergent integral."""
    assert proper_time_a0_integral_is_lambda4_over_64pi2()


def test_proper_time_a1_integral():
    """Lambda^2 Einstein-Hilbert scaling from an actual convergent integral."""
    assert proper_time_a1_integral_is_lambda2_over_32pi2()


def test_dirac_eh_coefficient_negative_definite():
    """kappa = -Lambda^2/(96 pi^2) exactly, sign computed in a fixed scheme."""
    assert dirac_induced_eh_coefficient_is_negative_definite()


def test_boson_statistics_flips_sign():
    """Statistics mutation: fermion -> boson flips the induced sign exactly."""
    assert boson_statistics_flips_induced_sign()


def test_pure_dirac_sign_definite():
    """N Dirac species preserve the sign: fixed scheme + pure Dirac => definite 1/G."""
    assert pure_dirac_content_is_sign_definite()


def test_gaussian_scheme_constant_differs():
    """Second regulator: same Lambda^2 power and sign, O(1) constant ratio sqrt(pi)/2."""
    assert gaussian_scheme_same_power_and_sign_different_constant()


def test_derived_matches_convenience_form():
    """1/G = N c Lambda^2 is now derived (c0 = 1/(6 pi) computed), not postulated."""
    assert derived_newton_scaling_matches_convenience_form()


def test_newton_sign_not_theorem_without_input():
    """Without the scheme/statistics inputs the sign of N c Lambda^2 is undecidable."""
    assert newton_sign_is_not_a_theorem_without_scheme_input()


def test_inverse_newton_linear_in_N():
    """1/G is additive over decoupled fermion species."""
    assert inverse_newton_is_linear_in_fermion_count()


def test_inverse_newton_cutoff_squared():
    """Doubling the cutoff quadruples the induced 1/G."""
    assert inverse_newton_scales_as_cutoff_squared()


def test_eh_action_dimensionless():
    """S_EH is dimensionless in natural units with 1/G ~ Lambda^2."""
    assert einstein_hilbert_action_is_dimensionless()


def test_inverse_newton_mass_squared():
    """[1/G] = L^-2 = mass^2, matching Lambda^2."""
    assert inverse_newton_has_mass_squared_dimension()


def test_cosmological_dimensionless():
    """S_cc with rho_vac ~ Lambda^4 is dimensionless."""
    assert cosmological_term_is_dimensionless()


def test_higher_curvature_cutoff_independent():
    """R^2-type terms need no positive power of Lambda: EH dominates at low curvature."""
    assert higher_curvature_terms_are_cutoff_independent()
