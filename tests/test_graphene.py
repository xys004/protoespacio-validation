from validators.graphene import (
    cross_term_is_imaginary,
    fermi_velocity_is_three_half_ta,
    gradient_modulus_squared_is_dirac,
    phi_vanishes_at_K,
)


def test_phi_vanishes_at_K():
    assert phi_vanishes_at_K()


def test_gradient_modulus_is_dirac():
    assert gradient_modulus_squared_is_dirac()


def test_cross_term_is_imaginary():
    assert cross_term_is_imaginary()


def test_fermi_velocity():
    assert fermi_velocity_is_three_half_ta()
