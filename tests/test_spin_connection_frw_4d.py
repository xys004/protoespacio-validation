from validators.spin_connection_frw_4d import (
    cartan_torsion_system_is_unique_frw,
    de_sitter_ricci_is_minus_twelve_H_squared,
    flat_limit_zero_connection_and_curvature_frw,
    frw_ricci_closed_form,
    frw_tetrad_reproduces_metric,
    frw_two_route_ricci_agree,
    omega_wedge_omega_is_nonzero_frw,
    opposite_signature_gives_plus_six,
    solved_connection_has_zero_torsion_frw,
    torsion_vanishes_for_perturbed_connection_frw,
    two_routes_agree_without_omega_wedge_omega,
)


def test_tetrad_reproduces_metric():
    """e^T eta e = diag(1, -a^2, -a^2, -a^2) with the repo eta = (+,-,-,-)."""
    assert frw_tetrad_reproduces_metric()


def test_cartan_system_unique_4d():
    """24x24 linsolve: exactly one solution, omega^{0i} = -adot dx^i, rest zero."""
    assert cartan_torsion_system_is_unique_frw()


def test_solved_connection_zero_torsion():
    """Round trip: the solved connection annihilates all torsion components."""
    assert solved_connection_has_zero_torsion_frw()


def test_omega_wedge_omega_nonzero():
    """Nonabelian: (omega ^ omega)^{12}_{xy} = -adot^2 != 0 (absent in 2D)."""
    assert omega_wedge_omega_is_nonzero_frw()


def test_frw_two_route_agree():
    """Cartan and Christoffel Ricci scalars agree for arbitrary a(t)."""
    assert frw_two_route_ricci_agree()


def test_frw_closed_form():
    """R = -6(addot/a + adot^2/a^2) in the repo conventions (+,-,-,-)."""
    assert frw_ricci_closed_form()


def test_opposite_signature_flips_sign():
    """g = diag(-1, a^2, a^2, a^2) gives +6(addot/a + adot^2/a^2): the sign is
    exactly the overall metric-sign convention."""
    assert opposite_signature_gives_plus_six()


def test_static_limit_flat():
    """a = const => zero connection, zero curvature."""
    assert flat_limit_zero_connection_and_curvature_frw()


def test_de_sitter_cross_check():
    """a = e^{Ht} => R = -12 H^2 exactly (both routes), repo conventions."""
    assert de_sitter_ricci_is_minus_twelve_H_squared()


def test_perturbed_connection_has_torsion():
    """Negative control: epsilon-perturbed boost component is NOT torsion-free."""
    assert not torsion_vanishes_for_perturbed_connection_frw()


def test_abelian_truncation_breaks_agreement():
    """Negative control: dropping omega ^ omega breaks the two-route agreement."""
    assert not two_routes_agree_without_omega_wedge_omega()
