from validators.puentes_grupos import (
    boost_generators_antihermitian,
    four_pi_spinor_rotation_is_identity,
    gamma_five_anticommutes_with_all_gammas,
    gamma_five_squares_to_identity,
    rotation_generators_hermitian,
    six_independent_generators,
    two_pi_spinor_rotation_is_minus_identity,
)


def test_two_pi_gives_minus_identity():
    assert two_pi_spinor_rotation_is_minus_identity()


def test_four_pi_gives_identity():
    assert four_pi_spinor_rotation_is_identity()


def test_rotation_generators_hermitian():
    assert rotation_generators_hermitian()


def test_boost_generators_antihermitian():
    assert boost_generators_antihermitian()


def test_six_independent():
    assert six_independent_generators()


def test_gamma5_anticommutes():
    assert gamma_five_anticommutes_with_all_gammas()


def test_gamma5_squared():
    assert gamma_five_squares_to_identity()
