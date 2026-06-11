from validators.simetrias_paso import (
    U_is_unitary_for_all_k,
    complex_conjugation_inverts_k,
    inverse_of_U_is_U_at_minus_k,
    sigma_y_chiral_parity,
)


def test_sigma_y_parity_chiral():
    assert sigma_y_chiral_parity()


def test_complex_conjugation():
    assert complex_conjugation_inverts_k()


def test_inverse_matches_k_minus():
    assert inverse_of_U_is_U_at_minus_k()


def test_unitarity_for_all_k():
    assert U_is_unitary_for_all_k()
