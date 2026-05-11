from validators.protoespacio_global import (
    hamiltonian_is_hermitian,
    infrared_limit_is_dirac,
    quasi_energy_is_brillouin_periodic,
    wilson_term_lifts_seven_doublers,
)


def test_hamiltonian_hermitian():
    assert hamiltonian_is_hermitian()


def test_infrared_limit():
    assert infrared_limit_is_dirac()


def test_brillouin_periodicity():
    assert quasi_energy_is_brillouin_periodic()


def test_wilson_lifts_doublers():
    assert wilson_term_lifts_seven_doublers()
