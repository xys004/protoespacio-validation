from validators.weyl_dispersion import (
    weyl_eigenvalues_are_pm_vp,
    weyl_squared_holds,
)


def test_weyl_squared():
    assert weyl_squared_holds()


def test_weyl_eigenvalues():
    assert weyl_eigenvalues_are_pm_vp()
