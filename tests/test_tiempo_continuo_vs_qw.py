from validators.tiempo_continuo_vs_qw import (
    discrete_unitary_norm_one,
    trotter_first_order_matches_additive,
    trotter_second_order_has_commutator,
)


def test_trotter_first_order():
    assert trotter_first_order_matches_additive()


def test_trotter_second_order():
    assert trotter_second_order_has_commutator()


def test_unitarity():
    assert discrete_unitary_norm_one()
