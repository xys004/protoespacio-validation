"""
Dispersion Weyl 3D:  H_W = v (sigma . p)  =>  H_W^2 = v^2 |p|^2 I_2.
Autovalores: +/- v |p|.

Sustenta:
- book/chapters/06_De_Grafeno_a_Weyl_Dirac_3D.tex
- book/chapters/07_Weyl_3D_a_Dirac_3p1_discreto.tex
- book/chapters/08_QW_3D_Weyl_a_Dirac_3p1.tex
"""
from __future__ import annotations

import sympy as sp

from validators.clifford import pauli_matrices


def weyl_hamiltonian():
    v = sp.symbols("v", positive=True)
    px, py, pz = sp.symbols("p_x p_y p_z", real=True)
    sx, sy, sz = pauli_matrices()
    H = v * (sx * px + sy * py + sz * pz)
    return H, (v, px, py, pz)


def weyl_squared_holds() -> bool:
    """H_W^2 = v^2 (p_x^2 + p_y^2 + p_z^2) I_2."""
    H, (v, px, py, pz) = weyl_hamiltonian()
    expected = v**2 * (px**2 + py**2 + pz**2) * sp.eye(2)
    return sp.simplify(H * H - expected) == sp.zeros(2, 2)


def weyl_eigenvalues_are_pm_vp() -> bool:
    """Autovalores de H_W son +/- v |p|."""
    H, (v, px, py, pz) = weyl_hamiltonian()
    p_norm = sp.sqrt(px**2 + py**2 + pz**2)
    eigvals = list(H.eigenvals().keys())
    expected = {v * p_norm, -v * p_norm}
    return {sp.simplify(e) for e in eigvals} == {sp.simplify(e) for e in expected}
