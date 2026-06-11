from validators.wilson_subsector import (
    doubler_gets_mass_two_r_at_k_pi,
    linearization_near_zero_is_pure_dirac,
    massless_mode_at_k_zero,
    wilson_squared_is_diagonal_at_corners,
)


def test_massless_at_zero():
    assert massless_mode_at_k_zero()


def test_doubler_mass_at_pi():
    assert doubler_gets_mass_two_r_at_k_pi()


def test_linearization_is_dirac():
    assert linearization_near_zero_is_pure_dirac()


def test_squared_at_corners():
    assert wilson_squared_is_diagonal_at_corners()
