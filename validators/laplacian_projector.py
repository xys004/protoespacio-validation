"""
Graph-Laplacian low-mode projectors for Paper II.

Paper II replaces small-momentum modes (which require a Brillouin torus)
by the low eigenmodes of a finite graph Laplacian L_G. This module builds
L_G on small finite graphs, certifies the exact cycle-graph spectrum, the
near-zero quadratic behaviour matching the continuum |k|^2 dispersion, and
the basic properties of the low-mode spectral projector P_Lambda.

Everything is exact in sympy: the spectral projector is assembled from the
exact eigenspaces (Gram--Schmidt orthonormalization), so P_Lambda^2 =
P_Lambda and P_Lambda = P_Lambda^dagger are checked as exact matrix
identities rather than numerically.

Sustains:
- paper2_graph_local_dirac_reconstruction_v2.tex
  (Definition "graph-local protospace datum"; section "Low modes without
   Bloch momentum"; subsection "Periodic chain as a sanity check")
"""
from __future__ import annotations

from collections.abc import Iterable

import sympy as sp


# --------------------------------------------------------------------------
# Exact symbolic graph Laplacians
# --------------------------------------------------------------------------

def cycle_shift(n: int) -> sp.Matrix:
    """Cyclic shift S on C_n, with S[x, x+1 mod n] = 1."""
    return sp.Matrix(n, n, lambda i, j: 1 if (j - i) % n == 1 else 0)


def cycle_laplacian(n: int) -> sp.Matrix:
    """Combinatorial Laplacian L = 2I - S - S^T of the cycle graph C_n."""
    s = cycle_shift(n)
    return 2 * sp.eye(n) - s - s.T


def graph_laplacian(n_vertices: int, edges: Iterable[tuple[int, int]]) -> sp.Matrix:
    """Combinatorial Laplacian L = D - A for an arbitrary finite simple graph."""
    a = sp.zeros(n_vertices, n_vertices)
    for i, j in edges:
        a[i, j] = 1
        a[j, i] = 1
    deg = sp.diag(*[sum(a.row(i)) for i in range(n_vertices)])
    return deg - a


# --------------------------------------------------------------------------
# Cycle-graph spectrum (exact)
# --------------------------------------------------------------------------

def cycle_laplacian_eigenvalues_exact(n: int) -> bool:
    """The spectrum of L(C_n) is {2 - 2 cos(2 pi k / n) : k = 0..n-1}."""
    eigs = cycle_laplacian(n).eigenvals()  # {value: multiplicity}
    spectrum: dict[sp.Expr, int] = {}
    for val, mult in eigs.items():
        spectrum[sp.simplify(val)] = spectrum.get(sp.simplify(val), 0) + mult
    target: dict[sp.Expr, int] = {}
    for k in range(n):
        val = sp.simplify(2 - 2 * sp.cos(2 * sp.pi * k / n))
        target[val] = target.get(val, 0) + 1
    return spectrum == target


def cycle_low_mode_quadratic(n: int = 12) -> bool:
    """Near k = 0 the cycle dispersion is quadratic: lambda(k) = k^2 + O(k^4).

    With lambda(k) = 2 - 2 cos(k), the Taylor expansion to fourth order is
    k^2 - k^4/12 + O(k^6); the leading term is exactly k^2.
    """
    k = sp.symbols("k", real=True)
    lam = 2 - 2 * sp.cos(k)
    series = sp.series(lam, k, 0, 5).removeO()
    leading = series.coeff(k, 2)
    quartic = series.coeff(k, 4)
    return leading == 1 and quartic == sp.Rational(-1, 12)


# --------------------------------------------------------------------------
# Low-mode spectral projector P_Lambda (exact, sympy)
# --------------------------------------------------------------------------

def _le_threshold(val: sp.Expr, threshold) -> bool:
    """Exact-then-numeric comparison val <= threshold for real algebraic val."""
    diff = sp.nsimplify(threshold) - val
    return bool(sp.simplify(diff) >= 0) or bool(sp.N(diff) >= -sp.Float("1e-30"))


def low_mode_projector(lap: sp.Matrix, threshold) -> sp.Matrix:
    """Exact spectral projector P_Lambda = 1_{[0, threshold]}(L).

    Assembled from the orthonormalized eigenspaces of L whose eigenvalue is
    at or below the threshold.
    """
    n = lap.shape[0]
    proj = sp.zeros(n, n)
    for val, _mult, vecs in lap.eigenvects():
        if not _le_threshold(sp.simplify(val), threshold):
            continue
        onb = sp.GramSchmidt([sp.Matrix(v) for v in vecs], orthonormal=True)
        for u in onb:
            proj += u * u.H
    return sp.simplify(proj)


def projector_is_idempotent_hermitian(lap: sp.Matrix, threshold) -> bool:
    """P_Lambda^2 = P_Lambda and P_Lambda = P_Lambda^dagger, exactly."""
    p = low_mode_projector(lap, threshold)
    idempotent = sp.simplify(p * p - p) == sp.zeros(*p.shape)
    hermitian = sp.simplify(p - p.H) == sp.zeros(*p.shape)
    return idempotent and hermitian


def low_mode_count(lap: sp.Matrix, threshold) -> int:
    """Number of Laplacian eigenvalues at or below threshold (with multiplicity)."""
    count = 0
    for val, mult in lap.eigenvals().items():
        if _le_threshold(sp.simplify(val), threshold):
            count += mult
    return count


def projector_rank_counts_low_modes(lap: sp.Matrix, threshold) -> bool:
    """rank(P_Lambda) equals the number of low Laplacian modes."""
    p = low_mode_projector(lap, threshold)
    return p.rank() == low_mode_count(lap, threshold)


def constant_mode_is_always_low(n: int = 8) -> bool:
    """The zero mode (constant vector) is always in the low-mode window.

    L(C_n) annihilates the all-ones vector, so for any threshold >= 0 the
    low-mode window has dimension at least one.
    """
    lap = cycle_laplacian(n)
    ones = sp.ones(n, 1)
    annihilated = sp.simplify(lap * ones) == sp.zeros(n, 1)
    return annihilated and low_mode_count(lap, 0) >= 1
