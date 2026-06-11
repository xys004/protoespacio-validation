from validators.spin_connection import (
    conformal_ricci_closed_form,
    flat_tetrad_has_zero_connection_and_curvature,
    tetrad_and_metric_ricci_agree,
    torsion_is_zero_2d,
)


def test_torsion_free():
    assert torsion_is_zero_2d()


def test_tetrad_metric_ricci_agree():
    assert tetrad_and_metric_ricci_agree()


def test_conformal_ricci_closed_form():
    assert conformal_ricci_closed_form()


def test_flat_limit():
    assert flat_tetrad_has_zero_connection_and_curvature()
