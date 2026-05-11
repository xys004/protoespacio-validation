"""
Espectro global del paso discreto: cuasienergia circular y coexistencia
de los sectores omega = 0 y omega = pi.

Para un paso unitario U(k) sobre la zona de Brillouin:
  - los autovalores de U(k) viven en el circulo unidad (|lambda| = 1);
  - la cuasienergia omega(k) se define por lambda = exp(i omega(k) Delta t)
    => omega in (-pi/Delta t, pi/Delta t] (intervalo circular);
  - en tiempo discreto coexisten autoenergias en omega = 0 (modos triviales)
    y omega = pi/Delta t (modos pi), ausentes en tiempo continuo.

Verificacion sympy con un walk 1D simple:
  U(k) = cos(k) I - i sin(k) sigma_z
        = exp(-i k sigma_z)

  - en k = 0:   U = I, autovalores {1, 1}, omega = 0.
  - en k = pi:  U = -I, autovalores {-1, -1}, omega = pi.
  - 2pi-periodica (cuasienergia circular).

Verificacion z3 de la clasificacion combinatoria (cap. 32):
  8 esquinas, 4 en omega = 0 y 4 en omega = pi, con cargas quirales
  +/- 1 sumando 0 (Nielsen-Ninomiya global).

Sustenta:
- book/chapters/32_Consistencia_Global_Brillouin.tex
- (rama espectral abierta declarada en la revision del 28-mar-2026)
"""
from __future__ import annotations

import sympy as sp
import z3

from validators.clifford import pauli_matrices


def _U_of_k():
    """U(k) = cos(k) I - i sin(k) sigma_z."""
    _sx, _sy, sz = pauli_matrices()
    k = sp.symbols("k", real=True)
    U = sp.cos(k) * sp.eye(2) - sp.I * sp.sin(k) * sz
    return U, k


def zero_mode_at_k_zero() -> bool:
    """En k = 0, los dos autovalores son 1 (cuasienergia 0)."""
    U, k = _U_of_k()
    eigs = U.subs(k, 0).eigenvals()
    return all(sp.simplify(e - 1) == 0 for e in eigs)


def pi_mode_at_k_pi() -> bool:
    """En k = pi, los dos autovalores son -1 (cuasienergia pi)."""
    U, k = _U_of_k()
    eigs = U.subs(k, sp.pi).eigenvals()
    return all(sp.simplify(e + 1) == 0 for e in eigs)


def quasi_energy_two_pi_periodic() -> bool:
    """U(k + 2 pi) = U(k): la cuasienergia vive en el circulo."""
    U, k = _U_of_k()
    diff = sp.simplify(U.subs(k, k + 2 * sp.pi) - U)
    return diff == sp.zeros(2, 2)


def spectrum_on_unit_circle_for_all_k() -> bool:
    """Para todo k real, los autovalores de U(k) tienen modulo 1.

    Como U(k) es manifiestamente unitaria (U^dagger U = I), basta verificar
    eso simbolicamente.
    """
    U, _k = _U_of_k()
    return sp.simplify(U.H * U - sp.eye(2)) == sp.zeros(2, 2)


# --- z3: clasificacion combinatoria de las 8 esquinas (cap 32)

def eight_corner_family_is_sat() -> bool:
    """8 esquinas con cuasienergia en {0, pi} y cargas quirales +/- 1,
    con 4 en cada sector de cuasienergia y suma chiral = 0. SAT."""
    s = z3.Solver()
    N = 8
    eps = [z3.Int(f"eps_{i}") for i in range(N)]
    chi = [z3.Int(f"chi_{i}") for i in range(N)]
    for e in eps:
        s.add(z3.Or(e == 0, e == 1))  # 0 = energia 0, 1 = energia pi
    for c in chi:
        s.add(z3.Or(c == 1, c == -1))
    s.add(z3.Sum([z3.If(e == 0, 1, 0) for e in eps]) == 4)
    s.add(z3.Sum([z3.If(e == 1, 1, 0) for e in eps]) == 4)
    s.add(z3.Sum(chi) == 0)
    return s.check() == z3.sat


def chirality_imbalance_unsat() -> bool:
    """Si pedimos 5 cargas +1 y 3 cargas -1 (imbalance = 2), Nielsen-Ninomiya
    falla. UNSAT."""
    s = z3.Solver()
    N = 8
    chi = [z3.Int(f"chi_{i}") for i in range(N)]
    for c in chi:
        s.add(z3.Or(c == 1, c == -1))
    pos = z3.Sum([z3.If(c == 1, 1, 0) for c in chi])
    s.add(pos == 5)
    s.add(z3.Sum(chi) == 0)
    return s.check() == z3.unsat


def all_positive_in_zero_sector_is_sat() -> bool:
    """Caso extremo: las 4 cargas +1 en sector omega=0, las 4 cargas -1 en
    sector omega=pi. Total cero, distribucion 4+4. SAT.
    """
    s = z3.Solver()
    N = 8
    eps = [z3.Int(f"eps_{i}") for i in range(N)]
    chi = [z3.Int(f"chi_{i}") for i in range(N)]
    for e in eps:
        s.add(z3.Or(e == 0, e == 1))
    for c in chi:
        s.add(z3.Or(c == 1, c == -1))
    # Forzamos: chi=+1 sii eps=0
    for i in range(N):
        s.add((chi[i] == 1) == (eps[i] == 0))
    s.add(z3.Sum([z3.If(e == 0, 1, 0) for e in eps]) == 4)
    s.add(z3.Sum(chi) == 0)
    return s.check() == z3.sat
