"""A variable tetrad from an explicit local discrete Hamiltonian.

This module keeps three logically different statements separate.

1. Discrete-to-continuum. On a one-dimensional link (and independently in
   each coordinate direction), put a real velocity on the midpoint of every
   nearest-neighbour edge and define

       (K_a psi)_n = -i/(2a) [v_{n+1/2} psi_{n+1}
                              - v_{n-1/2} psi_{n-1}].

   The finite matrix is Hermitian. Taylor expansion of the actual link
   operator gives

       K_a = -i [v d_x + (1/2) d_x v] + O(a^2),

   so a sum sigma^i K_{a,i} tends to (1/2){v_i,p_i} sigma^i. The gradient
   term is therefore derived from a discrete substrate, not inserted into an
   already-continuum Hamiltonian.

2. Frozen-symbol geometry. For a wave packet centred at x_0 with wavelength
   ell much shorter than the inhomogeneity scale L, freeze v_i(x_0). The
   principal symbol squares to sum_i v_i^2 p_i^2 I and identifies the inverse
   tetrad e_i^j = v_i delta_i^j. This is a statement about the principal
   symbol, not the square of the full variable-coefficient differential
   operator.

3. Connection. The link expansion reconstructs a tetrad and its Hermitian
   ordering term. It does not, by itself, prove that this term is the complete
   Levi-Civita spin connection. The torsion-free connection used downstream
   is the standard additional geometric choice fixed by the reconstructed
   tetrad. The last check below verifies that the conformal tetrad supplied
   here is exactly the input convention consumed by ``spin_connection.py``.

There are two small ratios, not one: a/ell controls the lattice continuum
error, while ell/L controls the adiabatic freezing error. Conflating both
with a/L hides the intermediate wave-packet scale.

Sustains:
- master_protospace.tex, Sec. "A variable tetrad from a discrete link operator"
- book/chapters/23_Triada_Variable_Fondo_Geometrico.tex (only after its wording
  is updated to the same scope)
"""
from __future__ import annotations

import sympy as sp

from validators.clifford import pauli_matrices


def _continuum_link_expression():
    """Return the centred link quotient before multiplication by -i."""
    x = sp.symbols("x", real=True)
    a = sp.symbols("a", positive=True)
    velocity = sp.Function("v")(x)
    wave = sp.Function("psi")(x)
    forward = velocity.subs(x, x + a / 2) * wave.subs(x, x + a)
    backward = velocity.subs(x, x - a / 2) * wave.subs(x, x - a)
    return x, a, velocity, wave, (forward - backward) / (2 * a)


def discrete_link_hamiltonian_is_hermitian(n: int = 5) -> bool:
    """The periodic finite-link matrix is Hermitian for arbitrary real links."""
    a = sp.symbols("a", positive=True, real=True)
    weights = sp.symbols(f"w0:{n}", real=True)
    operator = sp.zeros(n, n)
    for site in range(n):
        right = (site + 1) % n
        left = (site - 1) % n
        operator[site, right] += -sp.I * weights[site] / (2 * a)
        operator[site, left] += sp.I * weights[left] / (2 * a)
    return sp.simplify(operator - operator.adjoint()) == sp.zeros(n, n)


def discrete_link_continuum_limit_is_symmetrized() -> bool:
    """The exact a->0 limit is v psi' + (v'/2) psi."""
    x, a, velocity, wave, quotient = _continuum_link_expression()
    expected = velocity * sp.diff(wave, x) + sp.diff(velocity, x) * wave / 2
    numerator = sp.expand(2 * a * quotient)
    lhopital_limit = sp.diff(numerator, a).subs(a, 0) / 2
    return sp.simplify(lhopital_limit - expected) == 0


def centered_link_error_is_second_order() -> bool:
    """The first lattice correction is O(a^2), with its coefficient checked."""
    x, a, velocity, wave, quotient = _continuum_link_expression()
    leading = velocity * sp.diff(wave, x) + sp.diff(velocity, x) * wave / 2
    correction = (
        velocity * sp.diff(wave, x, 3) / 6
        + sp.diff(velocity, x) * sp.diff(wave, x, 2) / 4
        + sp.diff(velocity, x, 2) * sp.diff(wave, x) / 8
        + sp.diff(velocity, x, 3) * wave / 48
    )
    series = sp.series(quotient, a, 0, 4).removeO()
    return sp.simplify(series - leading - a**2 * correction) == 0


def squared_hamiltonian_gives_pointwise_metric() -> bool:
    """The frozen principal symbol squares to g^{ij} p_i p_j I.

    Momenta here are commuting symbol variables and velocities are frozen at a
    point. This deliberately does not claim that the full variable-coefficient
    differential operator has the same square.
    """
    sx, sy, sz = pauli_matrices()
    vx, vy, vz = sp.symbols("v_x v_y v_z", real=True)
    px, py, pz = sp.symbols("p_x p_y p_z", real=True)
    symbol = vx * sx * px + vy * sy * py + vz * sz * pz
    expected = (vx**2 * px**2 + vy**2 * py**2 + vz**2 * pz**2) * sp.eye(2)
    return sp.simplify(symbol * symbol - expected) == sp.zeros(2, 2)


def inverse_tetrad_reads_off_velocities() -> bool:
    """e_a^i = v_a delta_a^i gives g^{ij} = diag(v_x^2,v_y^2,v_z^2)."""
    x = sp.symbols("x", real=True)
    vx, vy, vz = sp.Function("v_x")(x), sp.Function("v_y")(x), sp.Function("v_z")(x)
    e_inv = sp.diag(vx, vy, vz)
    g_inv = e_inv.T * e_inv
    expected = sp.diag(vx**2, vy**2, vz**2)
    return sp.simplify(g_inv - expected) == sp.zeros(3, 3)


def homogeneous_step_is_flat() -> bool:
    """Constant link velocities give a constant tetrad and flat metric."""
    x = sp.symbols("x", real=True)
    v = sp.Symbol("v0", positive=True)
    e_inv = sp.diag(v, v, v)
    dx_e = e_inv.applyfunc(lambda component: sp.diff(component, x))
    return dx_e == sp.zeros(3, 3)


def graded_step_has_nonzero_tetrad_gradient() -> bool:
    """A nonconstant link velocity gives a nonzero tetrad gradient."""
    x = sp.symbols("x", real=True)
    vx = sp.Function("v_x")(x)
    return sp.diff(vx, x) != 0


def conformal_tetrad_matches_spin_connection_input() -> bool:
    """The reconstructed conformal inverse tetrad yields g=e^(2phi) delta.

    This is the explicit convention seam to ``spin_connection.py``. The
    torsion-free connection is then solved there; it is not inferred by this
    equality alone.
    """
    x, y = sp.symbols("x y", real=True)
    phi = sp.Function("phi")(x, y)
    inverse_tetrad = sp.exp(-phi) * sp.eye(2)
    inverse_metric = inverse_tetrad.T * inverse_tetrad
    metric = sp.simplify(inverse_metric.inv())
    return sp.simplify(metric - sp.exp(2 * phi) * sp.eye(2)) == sp.zeros(2, 2)
