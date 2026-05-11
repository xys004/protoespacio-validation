from validators.causality import (
    effective_metric_has_lorentzian_signature,
    group_velocity_below_v_for_massive,
    group_velocity_squared_minus_v_squared,
    isotropic_cone_determinant,
    massless_dispersion_saturates_cone,
)


def test_group_velocity_diff_formula():
    assert group_velocity_squared_minus_v_squared()


def test_massless_saturates_cone():
    assert massless_dispersion_saturates_cone()


def test_group_velocity_strictly_below_v():
    assert group_velocity_below_v_for_massive()


def test_lorentzian_signature():
    assert effective_metric_has_lorentzian_signature()


def test_cone_determinant():
    assert isotropic_cone_determinant()
