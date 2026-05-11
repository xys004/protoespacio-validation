from validators.clifford import clifford_holds, sigma_antisymmetric


def test_dirac_anticommutator():
    assert clifford_holds()


def test_sigma_mu_nu_antisymmetric():
    assert sigma_antisymmetric()
