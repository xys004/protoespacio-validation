from validators.spin_connection_general_2d import (
    conformal_case_reduces_to_hardcoded_connection,
    flat_limit_zero_connection_and_curvature_general_2d,
    general_two_route_ricci_agree,
    routes_agree_for_sign_flipped_connection_general_2d,
    solved_connection_has_zero_torsion_general_2d,
    torsion_free_connection_is_unique_general_2d,
    torsion_vanishes_for_perturbed_connection_general_2d,
)


def test_torsion_free_connection_unique():
    """linsolve exhibits exactly one solution: omega^{12} = (A_y/B) dx - (B_x/A) dy."""
    assert torsion_free_connection_is_unique_general_2d()


def test_solved_connection_zero_torsion():
    """Round trip: the solved connection makes T^1 = T^2 = 0 identically."""
    assert solved_connection_has_zero_torsion_general_2d()


def test_two_route_ricci_agree():
    """Cartan and metric/Christoffel Ricci scalars agree for arbitrary A, B."""
    assert general_two_route_ricci_agree()


def test_conformal_case_reduces_to_existing_module():
    """A = B = e^phi recovers the hardcoded omega = phi_y dx - phi_x dy."""
    assert conformal_case_reduces_to_hardcoded_connection()


def test_flat_limit():
    """Constant A, B => zero connection and zero curvature."""
    assert flat_limit_zero_connection_and_curvature_general_2d()


def test_perturbed_connection_has_torsion():
    """Negative control: an epsilon-perturbed connection is NOT torsion-free."""
    assert not torsion_vanishes_for_perturbed_connection_general_2d()


def test_sign_flipped_connection_breaks_agreement():
    """Negative control: sign-flipped connection breaks the two-route agreement."""
    assert not routes_agree_for_sign_flipped_connection_general_2d()
