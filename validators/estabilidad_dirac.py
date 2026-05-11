"""
Estabilidad espectral de Dirac/Weyl frente a perturbaciones controladas.

Verifica:
  (1) Para H_D = v tau_z (x) (sigma . p) + m tau_x (x) I (Dirac canonico 4D),
      la masa anticommuta con la parte cinetica, asi que H_D^2 = (v^2 |p|^2 + m^2) I_4.
      Gap = 2 m en p = 0.

  (2) En Weyl 3D H_W = v sigma . p (2 componentes), no existe matriz 2x2
      (de la base {I, sigma_x, sigma_y, sigma_z}) que anticommute con las
      tres direcciones simultaneamente. Por eso un nodo Weyl no admite
      termino de masa local: es estable.

Sustenta:
- book/chapters/13_Perturbaciones_Dirac_Weyl_3D.tex
- book/chapters/24_Deformacion_Regla_Paso.tex
"""
from __future__ import annotations

import sympy as sp

from validators.clifford import pauli_matrices


def dirac_mass_anticommutes_with_kinetic() -> bool:
    """{v tau_z (x) (sigma . p), m tau_x (x) I} = 0."""
    v, m = sp.symbols("v m", positive=True)
    px, py, pz = sp.symbols("p_x p_y p_z", real=True)
    sx, sy, sz = pauli_matrices()
    I2 = sp.eye(2)
    tau_z, tau_x = sz, sx
    sigma_dot_p = sx * px + sy * py + sz * pz
    kinetic = v * sp.kronecker_product(tau_z, sigma_dot_p)
    mass = m * sp.kronecker_product(tau_x, I2)
    return sp.simplify(kinetic * mass + mass * kinetic) == sp.zeros(4, 4)


def dirac_gap_is_two_m() -> bool:
    """En p = 0, los autovalores de H_D = m tau_x (x) I son +/- m, cada uno
    doblemente degenerado. Gap entre bandas = 2 m."""
    v, m = sp.symbols("v m", positive=True)
    px, py, pz = sp.symbols("p_x p_y p_z", real=True)
    sx, sy, sz = pauli_matrices()
    I2 = sp.eye(2)
    tau_z, tau_x = sz, sx
    sigma_dot_p = sx * px + sy * py + sz * pz
    H = v * sp.kronecker_product(tau_z, sigma_dot_p) + m * sp.kronecker_product(tau_x, I2)
    H0 = H.subs({px: 0, py: 0, pz: 0})
    eigs_with_mult = H0.eigenvals()
    distinct = {sp.simplify(e) for e in eigs_with_mult.keys()}
    return distinct == {m, -m}


def dirac_squared_form() -> bool:
    """H_D^2 = (v^2 |p|^2 + m^2) I_4."""
    v, m = sp.symbols("v m", positive=True)
    px, py, pz = sp.symbols("p_x p_y p_z", real=True)
    sx, sy, sz = pauli_matrices()
    I2 = sp.eye(2)
    tau_z, tau_x = sz, sx
    sigma_dot_p = sx * px + sy * py + sz * pz
    H = v * sp.kronecker_product(tau_z, sigma_dot_p) + m * sp.kronecker_product(tau_x, I2)
    expected = (v**2 * (px**2 + py**2 + pz**2) + m**2) * sp.eye(4)
    return sp.simplify(H * H - expected) == sp.zeros(4, 4)


def weyl_admits_no_local_mass() -> bool:
    """Para H_W = v sigma . p, no existe M = m_0 I + m_x sigma_x + m_y sigma_y + m_z sigma_z
    no nulo que anticommute con sigma_x, sigma_y y sigma_z simultaneamente.

    Verificacion algebraica directa:
      {sigma_i, sigma_j} = 2 delta_{ij} I  =>  {sigma_i, M} = 2 m_i I + 2 m_0 sigma_i
      Anticommutador cero para todo i exige: m_0 = 0 y m_x = m_y = m_z = 0.
    """
    m_0, m_x, m_y, m_z = sp.symbols("m_0 m_x m_y m_z", real=True)
    sx, sy, sz = pauli_matrices()
    I2 = sp.eye(2)
    M = m_0 * I2 + m_x * sx + m_y * sy + m_z * sz
    # Anticonmutadores
    a_x = sp.simplify(sx * M + M * sx)
    a_y = sp.simplify(sy * M + M * sy)
    a_z = sp.simplify(sz * M + M * sz)
    # Para que los tres sean cero: M = 0
    sols = sp.solve(
        [a_x[0, 0], a_x[1, 1], a_x[0, 1], a_x[1, 0],
         a_y[0, 0], a_y[1, 1], a_y[0, 1], a_y[1, 0],
         a_z[0, 0], a_z[1, 1], a_z[0, 1], a_z[1, 0]],
        [m_0, m_x, m_y, m_z],
        dict=True,
    )
    # Unica solucion: todos cero
    if not sols:
        return False
    sol = sols[0]
    return all(sol.get(s, 0) == 0 for s in (m_0, m_x, m_y, m_z))
