from validators.curvature_numerics import (
    de_sitter_numeric_matches_minus_12_h2,
    fd_convergence_is_second_order,
    fd_convergence_is_second_order_4d,
    fd_ricci_matches_conformal_closed_form,
    fd_ricci_sphere_patch_is_two,
    flat_metrics_give_zero_curvature,
    frozen_spatial_axis_breaks_4d_check_but_not_frw,
    frw_numeric_ricci_matches_symbolic,
    graded_step_curvature_matches_closed_form,
    isotropic_schwarzschild_4d_is_ricci_flat,
    mutated_first_order_stencil_converges_at_second_order,
    mutated_frozen_axis_weak_field_matches_symbolic,
    mutated_schwarzschild_exponent_is_ricci_flat,
    regge_deficit_recovers_sphere_curvature,
    weak_field_4d_has_all_mixed_second_derivatives,
    weak_field_4d_numeric_ricci_matches_symbolic,
)


def test_fd_ricci_matches_conformal_closed_form():
    """FD Ricci from metric samples matches R = -2 e^{-2phi} Delta phi."""
    assert fd_ricci_matches_conformal_closed_form()


def test_fd_convergence_is_second_order():
    """2D: halving h divides the error by 4 within [3.85, 4.15], 4 points."""
    assert fd_convergence_is_second_order()


def test_fd_ricci_sphere_patch_is_two():
    """Stereographic unit-sphere patch: numerical R = 2 everywhere."""
    assert fd_ricci_sphere_patch_is_two()


def test_regge_deficit_recovers_sphere_curvature():
    """Deficit angles of geodesic triangles give K = 1 on the sphere, 0 flat."""
    assert regge_deficit_recovers_sphere_curvature()


def test_graded_step_curvature_matches_closed_form():
    """Substrate graded step v(x,y) -> metric v^{-2} delta -> numerical R."""
    assert graded_step_curvature_matches_closed_form()


# --- 4D symmetric special cases (one-coordinate metrics) -------------------

def test_frw_numeric_ricci_matches_symbolic():
    """4D FRW numerical Ricci matches the repo's certified symbolic route.

    Symmetric special case: the metric depends on t alone, so this exercises
    the 4x4 index machinery but not mixed-derivative differencing.
    """
    assert frw_numeric_ricci_matches_symbolic()


def test_de_sitter_numeric_matches_minus_12_h2():
    """De Sitter anchor through the 4D numerical pipeline: R = -12 H^2."""
    assert de_sitter_numeric_matches_minus_12_h2()


# --- genuinely 4D runs: mixed derivatives exercised ------------------------

def test_weak_field_4d_has_all_mixed_second_derivatives():
    """Precondition: all six mixed partials d_mu d_nu phi are nonzero."""
    assert weak_field_4d_has_all_mixed_second_derivatives()


def test_weak_field_4d_numeric_ricci_matches_symbolic():
    """Conformastatic phi(t,x,y,z): 4D numerical R matches the symbolic route."""
    assert weak_field_4d_numeric_ricci_matches_symbolic()


def test_fd_convergence_is_second_order_4d():
    """4D convergence order verified, not inherited from the 2D run."""
    assert fd_convergence_is_second_order_4d()


def test_isotropic_schwarzschild_4d_is_ricci_flat():
    """Schwarzschild in isotropic coordinates: numerical R -> 0 at order 2."""
    assert isotropic_schwarzschild_4d_is_ricci_flat()


def test_frozen_spatial_axis_breaks_4d_check_but_not_frw():
    """Freezing x3 leaves FRW untouched but breaks the genuinely-4D metric."""
    assert frozen_spatial_axis_breaks_4d_check_but_not_frw()


# --- mutation controls: all must FAIL -------------------------------------

def test_mutated_first_order_stencil_rejected():
    """Wrong stencil order: forward differences collapse the ratio to ~2."""
    assert not mutated_first_order_stencil_converges_at_second_order()


def test_mutated_frozen_axis_weak_field_rejected():
    """Dropped mixed derivatives: freezing x3 breaks the 4D agreement."""
    assert not mutated_frozen_axis_weak_field_matches_symbolic()


def test_mutated_schwarzschild_exponent_rejected():
    """Wrong conformal exponent: the vacuum metric stops being Ricci-flat."""
    assert not mutated_schwarzschild_exponent_is_ricci_flat()


def test_flat_metrics_give_zero_curvature():
    """Flat 2D and 4D Minkowski data give machine-zero curvature."""
    assert flat_metrics_give_zero_curvature()
