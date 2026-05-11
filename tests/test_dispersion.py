from validators.dispersion import dirac_dispersion_holds


def test_dirac_dispersion_squared():
    assert dirac_dispersion_holds()
