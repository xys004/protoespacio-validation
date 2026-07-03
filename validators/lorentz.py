"""
Generadores Lorentz espinoriales Sigma^{mu nu} = (i/4)[gamma^mu, gamma^nu].

Verifica:
  - antisimetria  Sigma^{mu nu} = - Sigma^{nu mu}
  - covariancia   [Sigma^{rho sigma}, gamma^mu] = i (eta^{sigma mu} gamma^rho - eta^{rho mu} gamma^sigma)
  - cierre        [Sigma^{mu nu}, Sigma^{rho sigma}] = i (eta^{mu rho} Sigma^{nu sigma} - eta^{mu sigma} Sigma^{nu rho} - eta^{nu rho} Sigma^{mu sigma} + eta^{nu sigma} Sigma^{mu rho})

Sustenta:
- book/chapters/15_De_Bloques_a_Gamma_SL2C.tex
- book/chapters/16_Generadores_Lorentz.tex
"""
from __future__ import annotations

import sympy as sp

from validators.clifford import (
    dirac_gamma_matrices,
    minkowski_metric,
    sigma_mu_nu,
)


def covariance_holds() -> bool:
    """[Sigma^{rho sigma}, gamma^mu] = i (eta^{sigma mu} gamma^rho - eta^{rho mu} gamma^sigma).

    Convencion: J^{mu nu} actuando sobre vector V^rho da
    [J^{mu nu}, V^rho] = i (eta^{nu rho} V^mu - eta^{mu rho} V^nu).
    """
    g = dirac_gamma_matrices()
    eta = minkowski_metric()
    Z4 = sp.zeros(4, 4)
    for rho in range(4):
        for sig in range(4):
            S = sigma_mu_nu(rho, sig)
            for mu in range(4):
                lhs = S * g[mu] - g[mu] * S
                rhs = sp.I * (eta[sig, mu] * g[rho] - eta[rho, mu] * g[sig])
                if sp.simplify(lhs - rhs) != Z4:
                    return False
    return True


def lorentz_algebra_closes() -> bool:
    """[Sigma^{mu nu}, Sigma^{rho sigma}] = i (eta^{nu rho} Sigma^{mu sigma}
                                            - eta^{mu rho} Sigma^{nu sigma}
                                            - eta^{nu sigma} Sigma^{mu rho}
                                            + eta^{mu sigma} Sigma^{nu rho})."""
    eta = minkowski_metric()
    Z4 = sp.zeros(4, 4)
    for mu in range(4):
        for nu in range(4):
            S_mn = sigma_mu_nu(mu, nu)
            for rho in range(4):
                for sig in range(4):
                    S_rs = sigma_mu_nu(rho, sig)
                    lhs = S_mn * S_rs - S_rs * S_mn
                    rhs = sp.I * (
                        eta[nu, rho] * sigma_mu_nu(mu, sig)
                        - eta[mu, rho] * sigma_mu_nu(nu, sig)
                        - eta[nu, sig] * sigma_mu_nu(mu, rho)
                        + eta[mu, sig] * sigma_mu_nu(nu, rho)
                    )
                    if sp.simplify(lhs - rhs) != Z4:
                        return False
    return True
