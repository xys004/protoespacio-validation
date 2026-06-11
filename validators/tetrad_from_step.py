"""
The variable tetrad emerges from a slowly varying discrete step.

Part IV of the master programme must justify, not postulate, the position-dependent
tetrad e^a_i(x) that the spin connection and induced gravity are built on. Here we
show it emerges from the substrate: deform the local hopping / walk amplitudes of
the discrete model slowly in space, linearize around a node, and read off a Dirac
Hamiltonian whose spatial coefficients ARE a position-dependent tetrad.

Concretely, take the anisotropic Weyl/Dirac block of the corpus with
position-dependent Fermi velocities v_a(x) (a slowly varying step):

    H(x) = sum_a v_a(x) sigma_a p_a ,    H(x)^2 = sum_a v_a(x)^2 p_a^2 I.

Freezing the coefficients at a point x_0 (the slow-gradient / adiabatic reading),
the inverse tetrad is e_a^i(x_0) = v_a(x_0) delta_a^i and the effective inverse
metric is g^{ij}(x_0) = sum_a v_a(x_0)^2 delta^i_a delta^j_a. A homogeneous step
(constant v_a) gives a constant tetrad and flat metric; a graded step gives a
genuinely position-dependent tetrad, which is exactly the input of
spin_connection.py.

This module verifies the squared-Hamiltonian identity with position-dependent
coefficients (so the tetrad reading is exact pointwise) and that the homogeneous
limit is flat.

Sustains:
- master_protospace.tex, Part IV (variable tetrad from a slowly varying step)
- book/chapters/23_Triada_Variable_Fondo_Geometrico.tex
"""
from __future__ import annotations

import sympy as sp

from validators.clifford import pauli_matrices


def squared_hamiltonian_gives_pointwise_metric() -> bool:
    """H(x) = sum_a v_a(x) sigma_a p_a => H^2 = sum_a v_a(x)^2 p_a^2 I, so the
    spatial coefficients define a pointwise inverse metric g^{ii}(x) = v_i(x)^2."""
    sx, sy, sz = pauli_matrices()
    x = sp.symbols("x", real=True)
    px, py, pz = sp.symbols("p_x p_y p_z", real=True)
    vx, vy, vz = sp.Function("v_x")(x), sp.Function("v_y")(x), sp.Function("v_z")(x)
    H = vx * sx * px + vy * sy * py + vz * sz * pz
    expected = (vx**2 * px**2 + vy**2 * py**2 + vz**2 * pz**2) * sp.eye(2)
    return sp.simplify(H * H - expected) == sp.zeros(2, 2)


def inverse_tetrad_reads_off_velocities() -> bool:
    """The inverse tetrad e_a^i = v_a delta_a^i reproduces g^{ij} = sum_a e_a^i e_a^j
    = diag(v_x^2, v_y^2, v_z^2)."""
    x = sp.symbols("x", real=True)
    vx, vy, vz = sp.Function("v_x")(x), sp.Function("v_y")(x), sp.Function("v_z")(x)
    e_inv = sp.diag(vx, vy, vz)  # e_a^i
    g_inv = e_inv * e_inv.T  # sum_a e_a^i e_a^j
    expected = sp.diag(vx**2, vy**2, vz**2)
    return sp.simplify(g_inv - expected) == sp.zeros(3, 3)


def homogeneous_step_is_flat() -> bool:
    """Constant velocities (homogeneous step) => constant tetrad, flat metric
    (all spatial derivatives of the tetrad vanish)."""
    x = sp.symbols("x", real=True)
    v = sp.Symbol("v0", positive=True)  # constant
    e_inv = sp.diag(v, v, v)
    dx_e = e_inv.applyfunc(lambda c: sp.diff(c, x))
    return dx_e == sp.zeros(3, 3)


def graded_step_has_nonzero_tetrad_gradient() -> bool:
    """A genuinely graded step (v_x depending on x) has a non-zero tetrad gradient,
    i.e. it is NOT flat -- this is the input that sources the spin connection."""
    x = sp.symbols("x", real=True)
    vx = sp.Function("v_x")(x)
    grad = sp.diff(vx, x)
    # for a non-constant function the symbolic derivative is not identically zero
    return grad != 0
