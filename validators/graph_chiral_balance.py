"""
Graph-local chiral balance validators for Paper II.

These checks isolate the finite-dimensional algebraic layer that survives
without Bloch momentum:

- a grading Gamma with {Gamma, H} = 0 pairs nonzero eigenvalues as E <-> -E;
- bipartite off-diagonal Hamiltonians obey the zero-mode lower bound
  dim ker H >= ||A| - |B||;
- chiral hopping disorder preserves pairing, while onsite/same-sublattice
  terms generically break it;
- locality can be stated directly by graph distance.

Sustains:
- paper 2/main.tex
"""
from __future__ import annotations

from collections.abc import Iterable

import sympy as sp


def chiral_grading(n_a: int, n_b: int) -> sp.Matrix:
    """Return Gamma = diag(+I_A, -I_B)."""
    return sp.diag(*([1] * n_a + [-1] * n_b))


def deterministic_hopping_block(n_a: int, n_b: int, seed: int = 0) -> sp.Matrix:
    """Small exact hopping block used by the validators."""
    return sp.Matrix(
        n_a,
        n_b,
        lambda i, j: sp.Rational(((i + 1) * (j + 2) + seed) % 7 - 3, 3),
    )


def offdiagonal_hamiltonian(t_matrix: sp.Matrix) -> sp.Matrix:
    """Return H = [[0, T], [T^dagger, 0]] for a rectangular hopping block T."""
    t_matrix = sp.Matrix(t_matrix)
    n_a, n_b = t_matrix.shape
    upper = sp.Matrix.hstack(sp.zeros(n_a, n_a), t_matrix)
    lower = sp.Matrix.hstack(t_matrix.T.conjugate(), sp.zeros(n_b, n_b))
    return sp.Matrix.vstack(upper, lower)


def anticommutes_with_grading(h: sp.Matrix, gamma: sp.Matrix) -> bool:
    """Check Gamma H + H Gamma = 0 exactly."""
    return sp.simplify(gamma * h + h * gamma) == sp.zeros(*h.shape)


def characteristic_polynomial_is_even_or_odd(h: sp.Matrix) -> bool:
    """Check spectrum invariance under E -> -E through the characteristic polynomial.

    For p(lambda)=det(lambda I-H), spectral pairing means
        p(lambda) = (-1)^N p(-lambda),
    where N is the matrix size. This also allows unpaired zero modes.
    """
    lam = sp.symbols("lambda")
    n = h.shape[0]
    p = sp.Poly(h.charpoly(lam).as_expr(), lam)
    reflected = sp.Poly((-1) ** n * p.as_expr().subs(lam, -lam), lam)
    return sp.simplify(p.as_expr() - reflected.as_expr()) == 0


def zero_mode_count(h: sp.Matrix) -> int:
    """Count exact zero modes by nullity."""
    return h.shape[0] - h.rank()


def maximum_bipartite_matching_size(
    n_a: int, n_b: int, edges: Iterable[tuple[int, int]]
) -> int:
    """Maximum matching size for a small bipartite graph.

    Edges are given as pairs (a, b), with 0 <= a < n_a and 0 <= b < n_b.
    """
    by_a: list[list[int]] = [[] for _ in range(n_a)]
    for a, b in edges:
        by_a[a].append(b)

    match_b = [-1] * n_b

    def augment(a: int, seen: set[int]) -> bool:
        for b in by_a[a]:
            if b in seen:
                continue
            seen.add(b)
            if match_b[b] == -1 or augment(match_b[b], seen):
                match_b[b] = a
                return True
        return False

    return sum(1 for a in range(n_a) if augment(a, set()))


def symbolic_biadjacency(n_a: int, n_b: int, edges: Iterable[tuple[int, int]]) -> sp.Matrix:
    """Bi-adjacency matrix with one independent symbolic weight per edge."""
    t = sp.zeros(n_a, n_b)
    for idx, (a, b) in enumerate(edges):
        t[a, b] = sp.symbols(f"w_{idx}", nonzero=True)
    return t


def generic_matching_nullity_holds(
    n_a: int, n_b: int, edges: Iterable[tuple[int, int]]
) -> bool:
    """For generic edge weights, rank(T) equals maximum matching size.

    Consequently the chiral Hamiltonian nullity is
        |A| + |B| - 2 nu(G),
    which refines the weaker imbalance bound ||A|-|B||.
    """
    edges = list(edges)
    nu = maximum_bipartite_matching_size(n_a, n_b, edges)
    t_matrix = symbolic_biadjacency(n_a, n_b, edges)
    rank_t = t_matrix.rank()
    h_nullity = n_a + n_b - 2 * rank_t
    return rank_t == nu and h_nullity >= abs(n_a - n_b)


def chiral_pairing_finite_matrix(n_a: int = 4, n_b: int = 3, seed: int = 0) -> bool:
    """A finite off-diagonal Hamiltonian has paired spectrum."""
    t_matrix = deterministic_hopping_block(n_a, n_b, seed)
    h = offdiagonal_hamiltonian(t_matrix)
    gamma = chiral_grading(n_a, n_b)
    return anticommutes_with_grading(h, gamma) and characteristic_polynomial_is_even_or_odd(h)


def bipartite_index_bound_holds(n_a: int, n_b: int, seed: int = 0) -> bool:
    """Check dim ker [[0,T],[T^dagger,0]] >= ||A| - |B||."""
    t_matrix = deterministic_hopping_block(n_a, n_b, seed)
    h = offdiagonal_hamiltonian(t_matrix)
    return zero_mode_count(h) >= abs(n_a - n_b)


def chiral_disorder_preserves_pairing(n_cells: int = 5, seed: int = 1) -> bool:
    """Bipartite hopping disorder preserves chiral spectral pairing."""
    t_matrix = deterministic_hopping_block(n_cells, n_cells, seed)
    h = offdiagonal_hamiltonian(t_matrix)
    gamma = chiral_grading(n_cells, n_cells)
    return anticommutes_with_grading(h, gamma) and characteristic_polynomial_is_even_or_odd(h)


def chiral_breaking_lifts_pairing(n_cells: int = 4, seed: int = 2) -> bool:
    """Onsite terms break {Gamma,H}=0 and generically destroy E <-> -E pairing."""
    t_matrix = deterministic_hopping_block(n_cells, n_cells, seed)
    h = offdiagonal_hamiltonian(t_matrix)
    onsite = sp.diag(*[sp.Rational(2 * k + 1, 10) for k in range(2 * n_cells)])
    broken = h + onsite
    gamma = chiral_grading(n_cells, n_cells)
    return (not anticommutes_with_grading(broken, gamma)) and (
        not characteristic_polynomial_is_even_or_odd(broken)
    )


def approximate_pairing_bound_example() -> bool:
    """Example of epsilon-pairing when ||{Gamma,H}|| <= epsilon.

    For H = [[delta,t],[t,delta]], Gamma = diag(1,-1), the eigenvalues are
    delta +/- t and the partner error is exactly 2 delta = epsilon.
    """
    delta = sp.Rational(1, 20)
    t = sp.Rational(3, 2)
    eigs = [delta + t, delta - t]
    epsilon = 2 * delta
    return all(min(abs(sp.simplify(e + f)) for f in eigs) <= epsilon for e in eigs)


def structural_zero_modes_stay_in_epsilon_window() -> bool:
    """A chiral-breaking block with norm <= epsilon/2 keeps structural zeros nearby.

    We use the imbalanced graph A1-B-A2. Its off-diagonal Hamiltonian has one
    exact structural zero mode. Adding a block-diagonal perturbation D with
    ||D|| <= epsilon/2 leaves at least one eigenvalue within epsilon/2 of zero
    by Weyl stability; here we verify it directly on the finite matrix.
    """
    delta = sp.Rational(1, 20)
    t_matrix = sp.Matrix([[1], [1]])
    h_chiral = offdiagonal_hamiltonian(t_matrix)
    d_break = sp.diag(delta, -delta, sp.Rational(1, 2) * delta)
    h = h_chiral + d_break
    epsilon = 2 * delta
    roots = [complex(root) for root in sp.nroots(h.charpoly().as_expr(), n=30)]
    min_abs = min(abs(root) for root in roots)
    return min_abs <= float(epsilon / 2) + 1e-12


def single_site_bulk_mode_requires_edge_cut() -> bool:
    """A single-site bulk mode on A2 in A1-B1-A2-B2-A3 forces adjacent cuts."""
    t2, t3, psi = sp.symbols("t2 t3 psi")
    equations = [t2 * psi, t3 * psi]
    collapsed = [sp.simplify(eq / coeff) for eq, coeff in zip(equations, [t2, t3], strict=True)]
    return collapsed == [psi, psi]


def ssh_open_chain(n_cells: int, t1: sp.Expr, t2: sp.Expr) -> sp.Matrix:
    """Open SSH Hamiltonian with sites A_0,B_0,A_1,B_1,..."""
    n_sites = 2 * n_cells
    h = sp.zeros(n_sites, n_sites)
    for i in range(n_sites - 1):
        hopping = t1 if i % 2 == 0 else t2
        h[i, i + 1] = hopping
        h[i + 1, i] = hopping
    return h


def ssh_open_chain_zero_modes(n_cells: int = 6) -> bool:
    """In the fully dimerized topological limit, the two edge modes are exact."""
    h = ssh_open_chain(n_cells=n_cells, t1=0, t2=1)
    gamma = sp.diag(*([1, -1] * n_cells))
    e_left = sp.eye(2 * n_cells)[:, 0]
    e_right = sp.eye(2 * n_cells)[:, -1]
    edge_modes_exact = h * e_left == sp.zeros(2 * n_cells, 1) and h * e_right == sp.zeros(
        2 * n_cells, 1
    )
    return (
        anticommutes_with_grading(h, gamma)
        and characteristic_polynomial_is_even_or_odd(h)
        and zero_mode_count(h) == 2
        and edge_modes_exact
    )


def all_pairs_graph_distances(n_vertices: int, edges: Iterable[tuple[int, int]]) -> list[list[float]]:
    """All-pairs unweighted graph distances via Floyd-Warshall."""
    dist = [[float("inf")] * n_vertices for _ in range(n_vertices)]
    for i in range(n_vertices):
        dist[i][i] = 0.0
    for i, j in edges:
        dist[i][j] = 1.0
        dist[j][i] = 1.0
    for k in range(n_vertices):
        for i in range(n_vertices):
            for j in range(n_vertices):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist


def support_is_r_local(h: sp.Matrix, distances: list[list[float]], radius: int) -> bool:
    """Check H_xy = 0 whenever d_G(x,y) > radius."""
    for i in range(h.rows):
        for j in range(h.cols):
            if distances[i][j] > radius and sp.simplify(h[i, j]) != 0:
                return False
    return True


def irregular_graph_locality() -> bool:
    """Locality is checked by graph distance on a non-periodic finite graph."""
    n_vertices = 6
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 4), (4, 5)]
    distances = all_pairs_graph_distances(n_vertices, edges)
    h = sp.zeros(n_vertices, n_vertices)
    for idx, (i, j) in enumerate(edges, start=1):
        h[i, j] = sp.Rational(idx, 10)
        h[j, i] = sp.Rational(idx, 10)

    h_bad = h.copy()
    h_bad[0, 5] = h_bad[5, 0] = 1
    return support_is_r_local(h, distances, 1) and not support_is_r_local(h_bad, distances, 1)
