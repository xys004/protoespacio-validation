from validators.lichnerowicz import (
    bianchi_constraint_count_is_one,
    bianchi_defect_is_pure_gamma5,
    contraction_is_scalar_multiple_of_identity,
    dsquared_curvature_coefficient_is_plus_one_quarter,
    flat_limit_removes_curvature_term,
    general_riemann_contraction_is_minus_two_R,
    general_riemann_reduces_to_ansatz,
    lichnerowicz_E_sign,
    mutated_bianchi_broken_matches_minus_two_R,
    mutated_coefficient_one_eighth_matches,
    mutated_coefficient_one_half_matches,
    raw_coefficient_is_minus_one_quarter,
    ricci_scalar_of_ansatz,
)


def test_contraction_scalar():
    """X on the maximally symmetric ansatz is a scalar multiple of I_4."""
    assert contraction_is_scalar_multiple_of_identity()


def test_ricci_of_ansatz():
    """R = 12 K for the maximally symmetric ansatz in d = 4."""
    assert ricci_scalar_of_ansatz()


def test_raw_coefficient_is_minus_one_quarter():
    """Signed: (gamma nabla)^2 = nabla^2 - R/4 (no absolute value)."""
    assert raw_coefficient_is_minus_one_quarter()


def test_E_sign_is_plus_one():
    """Computed heat-kernel endomorphism sign: E = +R/4 via i^2 = -1."""
    assert lichnerowicz_E_sign() == 1


def test_flat_limit():
    """K = 0 removes the curvature term (flat cone limit)."""
    assert flat_limit_removes_curvature_term()


def test_bianchi_constraint_count_is_one():
    """The 4^4 cyclic sums reduce to exactly one constraint: 20 free components."""
    assert bianchi_constraint_count_is_one()


def test_general_riemann_contraction_minus_two_R():
    """X = -2R I_4 for ARBITRARY Riemann (Weyl and traceless Ricci decouple)."""
    assert general_riemann_contraction_is_minus_two_R()


def test_general_riemann_reduces_to_ansatz():
    """The general certificate specializes to X = -24 K I_4, R = 12 K."""
    assert general_riemann_reduces_to_ansatz()


def test_dsquared_coefficient_plus_one_quarter():
    """D^2 = -nabla^2 + R/4 for arbitrary Riemann: E = +R/4, signed."""
    assert dsquared_curvature_coefficient_is_plus_one_quarter()


def test_mutation_coefficient_one_half_fails():
    """Genuine mutation: coefficient 1/2 injected into the contraction -> fails."""
    assert not mutated_coefficient_one_half_matches()


def test_mutation_coefficient_one_eighth_fails():
    """Genuine mutation: coefficient 1/8 injected into the contraction -> fails."""
    assert not mutated_coefficient_one_eighth_matches()


def test_mutation_broken_bianchi_fails():
    """Genuine mutation: dropping first Bianchi breaks X = -2R."""
    assert not mutated_bianchi_broken_matches_minus_two_R()


def test_bianchi_defect_is_pure_gamma5():
    """The Bianchi-broken defect is a pure gamma5 multiple of R_[abcd]."""
    assert bianchi_defect_is_pure_gamma5()
