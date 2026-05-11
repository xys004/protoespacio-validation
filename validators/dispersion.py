"""
Validacion de la dispersion Dirac efectiva:

  H_D = v tau_z (sigma . p) + m tau_x  =>  H_D^2 = (v^2 |p|^2 + m^2) I_4.

Sustenta:
- book/frontmatter/02_notacion_y_supuestos.tex (eq HDcanon)
- book/chapters/0X_Derivacion_Grafeno_a_Dirac.tex
"""
from __future__ import annotations

import sympy as sp

from validators.clifford import pauli_matrices


def dirac_hamiltonian_block() -> tuple[sp.Matrix, dict]:
    """H_D = v tau_z otimes (sigma . p) + m tau_x otimes I_2.

    Devuelve (H_D, simbolos).
    """
    v, m = sp.symbols("v m", positive=True)
    px, py, pz = sp.symbols("p_x p_y p_z", real=True)

    sx, sy, sz = pauli_matrices()
    I2 = sp.eye(2)
    tau_x = sx
    tau_z = sz

    sigma_dot_p = sx * px + sy * py + sz * pz
    H = v * sp.kronecker_product(tau_z, sigma_dot_p) + m * sp.kronecker_product(tau_x, I2)
    syms = dict(v=v, m=m, p=(px, py, pz))
    return H, syms


def dirac_dispersion_holds() -> bool:
    """Verifica H_D^2 = (v^2 |p|^2 + m^2) I_4 simbolicamente."""
    H, s = dirac_hamiltonian_block()
    v, m = s["v"], s["m"]
    px, py, pz = s["p"]
    expected = (v ** 2 * (px ** 2 + py ** 2 + pz ** 2) + m ** 2) * sp.eye(4)
    return sp.simplify(H * H - expected) == sp.zeros(4, 4)
