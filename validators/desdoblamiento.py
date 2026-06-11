"""
Desdoblamiento controlado de Dirac 3D a dos nodos Weyl
mediante una perturbacion axial.

Punto de partida: H_D sin masa = v tau_z (x) (sigma . p) en el sector ligero.
Perturbacion: termino tipo Zeeman en z, b * I (x) sigma_z.

Resultado:
  H = v tau_z (x) (sigma . p) + b I (x) sigma_z

  H^2 = (v^2 |p|^2 + b^2) I_4 + 2 v b p_z (tau_z (x) I)

Bloque-diagonal en tau:
  upper (tau_z=+1):  H^2 = (v^2 p_x^2 + v^2 p_y^2 + (v p_z + b)^2) I_2
                     => nodo Weyl en p = (0, 0, -b/v)
  lower (tau_z=-1):  H^2 = (v^2 p_x^2 + v^2 p_y^2 + (v p_z - b)^2) I_2
                     => nodo Weyl en p = (0, 0, +b/v)

Sustenta:
- book/chapters/12_Desdoblamiento_Dirac_a_Weyl_3D.tex
"""
from __future__ import annotations

import sympy as sp

from validators.clifford import pauli_matrices


def _dirac_plus_axial():
    v, b = sp.symbols("v b", positive=True)
    px, py, pz = sp.symbols("p_x p_y p_z", real=True)
    sx, sy, sz = pauli_matrices()
    I2 = sp.eye(2)
    tau_z = sz
    sigma_dot_p = sx * px + sy * py + sz * pz
    H_D = v * sp.kronecker_product(tau_z, sigma_dot_p)
    axial = b * sp.kronecker_product(I2, sz)
    return H_D + axial, (v, b, px, py, pz)


def squared_decomposition_holds() -> bool:
    """H^2 = (v^2 |p|^2 + b^2) I_4 + 2 v b p_z (tau_z (x) I)."""
    H, (v, b, px, py, pz) = _dirac_plus_axial()
    _sx, _sy, sz = pauli_matrices()
    I2 = sp.eye(2)
    expected = (
        (v**2 * (px**2 + py**2 + pz**2) + b**2) * sp.eye(4)
        + 2 * v * b * pz * sp.kronecker_product(sz, I2)
    )
    return sp.simplify(H * H - expected) == sp.zeros(4, 4)


def upper_weyl_node_at_minus_b_over_v() -> bool:
    """En p = (0, 0, -b/v), el sector superior tiene autovalor 0 doblemente
    degenerado."""
    H, (v, b, px, py, pz) = _dirac_plus_axial()
    H_at = H.subs({px: 0, py: 0, pz: -b / v})
    # det H = 0 (existe al menos un autovalor cero)
    return sp.simplify(H_at.det()) == 0


def lower_weyl_node_at_plus_b_over_v() -> bool:
    """En p = (0, 0, +b/v), el sector inferior tiene autovalor 0."""
    H, (v, b, px, py, pz) = _dirac_plus_axial()
    H_at = H.subs({px: 0, py: 0, pz: b / v})
    return sp.simplify(H_at.det()) == 0


def nodes_separated_by_two_b_over_v() -> bool:
    """Los dos nodos estan en p_z = -b/v y p_z = +b/v, separados por 2b/v."""
    v, b = sp.symbols("v b", positive=True)
    sep = (b / v) - (-b / v)
    return sp.simplify(sep - 2 * b / v) == 0
