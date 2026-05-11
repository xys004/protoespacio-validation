from validators.lorentz import covariance_holds, lorentz_algebra_closes


def test_sigma_mu_nu_covariance():
    """[Sigma^{rho sigma}, gamma^mu] = i (eta^{rho mu} gamma^sigma - eta^{sigma mu} gamma^rho)."""
    assert covariance_holds()


def test_lorentz_algebra_closure():
    """Cierre del algebra de Lorentz espinorial Sigma^{mu nu}."""
    assert lorentz_algebra_closes()
