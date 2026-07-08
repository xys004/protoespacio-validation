from validators.curvature_numerics import (
    de_sitter_numeric_matches_minus_12_h2,
    fd_convergence_is_second_order,
    fd_ricci_matches_conformal_closed_form,
    fd_ricci_sphere_patch_is_two,
    flat_metrics_give_zero_curvature,
    frw_numeric_ricci_matches_symbolic,
    graded_step_curvature_matches_closed_form,
    regge_deficit_recovers_sphere_curvature,
)


def test_fd_ricci_matches_conformal_closed_form():
    """FD Ricci from metric samples matches R = -2 e^{-2phi} Delta phi."""
    assert fd_ricci_matches_conformal_closed_form()


def test_fd_convergence_is_second_order():
    """Halving h divides the error by ~4: controlled second-order limit."""
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


def test_frw_numeric_ricci_matches_symbolic():
    """4D FRW numerical Ricci matches the repo's certified symbolic route."""
    assert frw_numeric_ricci_matches_symbolic()


def test_de_sitter_numeric_matches_minus_12_h2():
    """De Sitter anchor through the 4D numerical pipeline: R = -12 H^2."""
    assert de_sitter_numeric_matches_minus_12_h2()


def test_flat_metrics_give_zero_curvature():
    """Flat 2D and 4D Minkowski data give machine-zero curvature."""
    assert flat_metrics_give_zero_curvature()
