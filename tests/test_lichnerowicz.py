from validators.lichnerowicz import (
    coefficient_is_not_one_half_or_one_eighth,
    contraction_is_scalar_multiple_of_identity,
    flat_limit_removes_curvature_term,
    lichnerowicz_coefficient_is_one_quarter,
    ricci_scalar_of_ansatz,
)


def test_contraction_scalar():
    assert contraction_is_scalar_multiple_of_identity()


def test_ricci_of_ansatz():
    assert ricci_scalar_of_ansatz()


def test_coefficient_one_quarter():
    assert lichnerowicz_coefficient_is_one_quarter()


def test_flat_limit():
    assert flat_limit_removes_curvature_term()


def test_coefficient_sharpness():
    assert coefficient_is_not_one_half_or_one_eighth()
