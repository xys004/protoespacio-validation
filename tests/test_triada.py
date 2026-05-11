from validators.triada import (
    diagonal_tetrad_2d_anisotropic,
    diagonal_tetrad_3d_gives_diagonal_metric,
    isotropic_limit_recovers_cone,
    position_dependent_tetrad_is_local,
)


def test_diagonal_tetrad_3d():
    assert diagonal_tetrad_3d_gives_diagonal_metric()


def test_diagonal_tetrad_2d():
    assert diagonal_tetrad_2d_anisotropic()


def test_isotropic_cone():
    assert isotropic_limit_recovers_cone()


def test_position_dependent_tetrad():
    assert position_dependent_tetrad_is_local()
