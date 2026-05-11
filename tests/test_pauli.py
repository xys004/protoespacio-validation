from validators.pauli import (
    anticommutator_holds,
    commutator_holds,
    levi_civita,
    squares_to_identity,
    traceless,
)


def test_pauli_anticommutator():
    assert anticommutator_holds()


def test_pauli_commutator():
    assert commutator_holds()


def test_pauli_squares():
    assert squares_to_identity()


def test_pauli_traceless():
    assert traceless()


def test_levi_civita_helper():
    assert levi_civita(0, 1, 2) == 1
    assert levi_civita(1, 0, 2) == -1
    assert levi_civita(0, 0, 1) == 0
    assert levi_civita(2, 0, 1) == 1
