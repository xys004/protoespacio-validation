from validators.heat_kernel_s2 import (
    dirac_a0_counts_spinor_components,
    dirac_a1_is_tr_R6_minus_E,
    dirac_spectrum_pins_E_to_plus_R_over_4,
    em_truncation_is_stable,
    mutated_halved_dirac_multiplicity_matches,
    mutated_scalar_degeneracy_matches,
    mutated_scalar_spectrum_matches,
    mutated_scalar_spectrum_on_spinors_matches,
    round_s2_area_is_4_pi_r2,
    round_s2_ricci_scalar_is_2_over_r2,
    scalar_a0_is_weyl_volume_term,
    scalar_a1_is_R_over_6,
    scalar_a2_matches_gilkey_formula,
)


def test_s2_ricci_scalar():
    """R = +2/r^2 from the metric, repo Christoffel/Riemann conventions."""
    assert round_s2_ricci_scalar_is_2_over_r2()


def test_s2_area():
    """Area = 4 pi r^2, computed from sqrt(g)."""
    assert round_s2_area_is_4_pi_r2()


def test_scalar_a0():
    """Leading Weyl term Area/(4 pi s): a_0 density = 1."""
    assert scalar_a0_is_weyl_volume_term()


def test_scalar_a1_R_over_6():
    """The universal Seeley-DeWitt 1/6, with sign, derived from the exact spectrum."""
    assert scalar_a1_is_R_over_6()


def test_scalar_a2_gilkey():
    """Spectral s^2 coefficient equals Gilkey's a_2 combination on S^2 data."""
    assert scalar_a2_matches_gilkey_formula()


def test_dirac_a0():
    """Dirac leading term counts the 2 spinor components (tr I_2 = 2 in d=2)."""
    assert dirac_a0_counts_spinor_components()


def test_dirac_a1_gilkey_combination():
    """Dirac a_1 density = tr[(R/6 - R/4) I_2] = -R/6, the induced-gravity input."""
    assert dirac_a1_is_tr_R6_minus_E()


def test_dirac_pins_lichnerowicz_E():
    """The exact Dirac spectrum forces E = +R/4 uniquely (sign included)."""
    assert dirac_spectrum_pins_E_to_plus_R_over_4()


def test_em_truncation_stability():
    """3, 4, 5 Bernoulli terms give identical series through s^2."""
    assert em_truncation_is_stable()


def test_mutation_wrong_scalar_spectrum():
    """Shifted eigenvalues (l+1/2)^2 break the a_1 = R/6 check (a_1 comes out R/24)."""
    assert not mutated_scalar_spectrum_matches()


def test_mutation_wrong_scalar_degeneracy():
    """Degeneracy 2l+2 breaks the structure check (sqrt(s) contamination)."""
    assert not mutated_scalar_degeneracy_matches()


def test_mutation_halved_dirac_multiplicity():
    """Forgetting the +/- doubling (4m -> 2m) breaks the Dirac structure check."""
    assert not mutated_halved_dirac_multiplicity_matches()


def test_mutation_scalar_spectrum_on_spinors():
    """Scalar eigenvalues on the spinor bundle break the Dirac structure check."""
    assert not mutated_scalar_spectrum_on_spinors_matches()
