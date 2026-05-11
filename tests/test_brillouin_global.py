from validators.brillouin_global import (
    all_positive_in_zero_sector_is_sat,
    chirality_imbalance_unsat,
    eight_corner_family_is_sat,
    pi_mode_at_k_pi,
    quasi_energy_two_pi_periodic,
    spectrum_on_unit_circle_for_all_k,
    zero_mode_at_k_zero,
)


# --- sympy

def test_zero_mode_at_origin():
    assert zero_mode_at_k_zero()


def test_pi_mode_at_k_pi():
    assert pi_mode_at_k_pi()


def test_quasi_energy_is_two_pi_periodic():
    assert quasi_energy_two_pi_periodic()


def test_spectrum_on_unit_circle():
    assert spectrum_on_unit_circle_for_all_k()


# --- z3

def test_eight_corners_satisfiable():
    assert eight_corner_family_is_sat()


def test_chirality_imbalance_unsat():
    assert chirality_imbalance_unsat()


def test_extreme_distribution_satisfiable():
    assert all_positive_in_zero_sector_is_sat()
