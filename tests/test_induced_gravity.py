from validators.induced_gravity import (
    a1_uses_operator_derived_sign,
    any_pure_fermion_content_is_sign_definite,
    boson_vacuum_energy_is_positive,
    conformal_coupling_kills_EH_term,
    conformal_scalar_gives_exactly_zero,
    cosmological_term_is_dimensionless,
    derived_newton_scaling_matches_convenience_form,
    dirac_a1_density_is_minus_R_over_12,
    dirac_induced_eh_coefficient_is_negative_definite,
    einstein_hilbert_action_is_dimensionless,
    einstein_hilbert_dominates_a2_by_inverse_cutoff_curvature_squared,
    einstein_hilbert_term_is_generated,
    field_content_kappas_match_closed_forms,
    gaussian_scheme_same_power_and_sign_different_constant,
    higher_curvature_terms_are_cutoff_independent,
    induced_sign_is_statistics_times_nonminimal_coupling,
    inverse_newton_has_mass_squared_dimension,
    inverse_newton_is_linear_in_fermion_count,
    inverse_newton_scales_as_cutoff_squared,
    minimal_scalar_and_dirac_add_rather_than_cancel,
    mutated_rank_four_boson_kappa_matches_a_physical_field,
    mutated_statistics_flip_predicts_scalar_sign,
    mutated_vacuum_energy_uses_a1_density,
    newton_sign_is_not_a_theorem_without_scheme_input,
    proper_time_a0_integral_is_lambda4_over_64pi2,
    proper_time_a1_integral_is_lambda2_over_32pi2,
    pure_dirac_content_is_sign_definite,
    pure_fermion_vacuum_energy_is_negative,
    sign_flip_requires_nonminimal_boson_not_statistics,
    total_sign_is_a_condition_on_matter_content,
    vacuum_energy_magnitude_lies_in_the_induced_scales_window,
    vacuum_energy_sign_is_statistics_alone_unlike_newton,
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


def test_field_content_table():
    """The five physical field contents reproduce their exact closed-form kappas."""
    assert field_content_kappas_match_closed_forms()


def test_sign_mechanism_is_statistics_times_coupling():
    """sign(kappa) = sign(sigma) * sign(xi_eff - 1/6): the corrected mechanism."""
    assert induced_sign_is_statistics_times_nonminimal_coupling()


def test_minimal_scalar_and_dirac_add():
    """Minimal scalar and Dirac both give kappa < 0: statistics does NOT cancel them."""
    assert minimal_scalar_and_dirac_add_rather_than_cancel()


def test_conformal_scalar_zero():
    """A conformally coupled scalar induces exactly zero Einstein-Hilbert term."""
    assert conformal_scalar_gives_exactly_zero()


def test_sign_flip_needs_nonminimal_boson():
    """The only kappa > 0 row is the xi = 1/4 scalar: flipping needs coupling, not statistics."""
    assert sign_flip_requires_nonminimal_boson_not_statistics()


def test_any_pure_fermion_content_sign_definite():
    """Arbitrary Weyl + Dirac content is sign-definite: xi_eff = 1/4 is forced for spin-1/2."""
    assert any_pure_fermion_content_is_sign_definite()


def test_total_sign_is_condition_on_content():
    """Mixed content realizes negative, exactly zero and positive totals: a condition, not a theorem."""
    assert total_sign_is_a_condition_on_matter_content()


def test_mutated_statistics_flip_mispredicts_scalar():
    """MUTATION: the discarded mechanism mispredicts the minimal scalar's kappa."""
    assert not mutated_statistics_flip_predicts_scalar_sign()


def test_mutated_rank_four_boson_matches_nothing():
    """MUTATION: the old validator's +Lambda^2/(96 pi^2) equals no physical field's kappa."""
    assert not mutated_rank_four_boson_kappa_matches_a_physical_field()


def test_pure_fermion_vacuum_energy_negative():
    """rho_vac < 0 for pure fermion content: -Lambda^4/(16 pi^2) per Dirac species."""
    assert pure_fermion_vacuum_energy_is_negative()


def test_boson_vacuum_energy_positive():
    """rho_vac > 0 for a real scalar: the vacuum-energy sign does flip with statistics."""
    assert boson_vacuum_energy_is_positive()


def test_vacuum_sign_is_statistics_only():
    """All three scalars share one rho_vac while their kappas differ in sign: a_0 carries no E."""
    assert vacuum_energy_sign_is_statistics_alone_unlike_newton()


def test_vacuum_magnitude_in_induced_scales_window():
    """The computed |rho_vac|/Lambda^4 = 1/(16 pi^2) lies inside induced_scales.py's c' window."""
    assert vacuum_energy_magnitude_lies_in_the_induced_scales_window()


def test_mutated_vacuum_from_a1():
    """MUTATION: building rho_vac from a_1 instead of a_0 splits the three scalars."""
    assert not mutated_vacuum_energy_uses_a1_density()


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
    """The a_2 proper-time term integrates to a pure log: no positive cutoff power."""
    assert higher_curvature_terms_are_cutoff_independent()


def test_eh_dominates_a2_quantitatively():
    """(a_2 term)/(a_1 term) = 2 log(Lambda L)/(Lambda L)^2: dominance as an inequality."""
    assert einstein_hilbert_dominates_a2_by_inverse_cutoff_curvature_squared()
