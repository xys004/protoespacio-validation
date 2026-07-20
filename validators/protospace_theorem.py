"""
A proposition that is proved *because of* the protospace datum.

MOTIVATION (what this module is for)
------------------------------------
Definition 1 of the master manuscript introduces the protospace tuple

    P = (G, F, H, D, Gamma, L_G, G_micro).

The manuscript's defence of that definition is a claim about *statability*:
the tuple is said to be the minimal data on which the construction's
structural results "can even be stated without a Brillouin torus". A
referee's standing objection is that statability is strictly weaker than
derivability, and that the paper's one candidate result -- the nullity
bound dim ker H = |A| + |B| - 2 rank T >= ||A| - |B|| -- is rank-nullity
for a bipartite Hamiltonian and consumes only {Gamma, H}. The graph, the
fiber, the Laplacian and the microscopic symmetries play no role in its
proof.

This module states and certifies a proposition that consumes more of the
tuple, together with the sharpness objects that show the extra entries are
load-bearing rather than decorative.

CONVENTIONS
-----------
* G = (V, E) is a finite simple connected graph, |V| = n.
* F = C^2 is the internal fiber; the Hilbert space is Hi = l^2(V) (x) F,
  with the tensor ordering (vertex, fiber), i.e. operators are
  kron(X_V, M_F).
* L_G = D - A is the combinatorial graph Laplacian on l^2(V), and
  A := L_G (x) I_F is its lift to Hi.
* Gamma := I_V (x) sigma_z is the fiber grading.
* The Hamiltonians used here are the graded (chiral) ones
      H = T (x) sigma_+ + T^dagger (x) sigma_-,
  with T an operator on l^2(V) supported on E union the diagonal. Then
  {Gamma, H} = 0 exactly, and H^2 = T T^dagger (x) diag(1,0)
  + T^dagger T (x) diag(0,1).
* ||.||_F is the Hilbert-Schmidt (Frobenius) norm. All bounds below are
  proved in ||.||_F, where the Sylvester constant is exactly 1. Since
  ||X||_op <= ||X||_F, every bound also holds with the operator norm on
  the left-hand side, which is the norm used in the manuscript's cone
  criterion. The reverse replacement is NOT valid and is not claimed.
* Norms are compared in squared (rational) form wherever possible, so the
  certificates are exact sympy identities and rational inequalities with
  no floating-point tolerance.

THE PROPOSITION (quotable form)
-------------------------------
Proposition (Infrared cone inheritance).
Let P = (G, F, Hi, D, Gamma, L_G, G_micro) be a protospace with G a finite
connected graph on n vertices, F ~ C^2, Hi = l^2(V) (x) F, D given by a
self-adjoint H on Hi, and Gamma = I (x) sigma_z a grading with
{Gamma, H} = 0. Put A := L_G (x) I_F. Fix a threshold Lambda >= 0 and a
candidate velocity v > 0, and set

    P_L  := 1_{[0,Lambda]}(L_G) (x) I_F,          Q_L := 1 - P_L,
    N_L  := rank 1_{[0,Lambda]}(L_G)              (infrared window dimension),
    g_L  := min{ mu - lam : lam, mu in spec(L_G), lam <= Lambda < mu }
                                                  (infrared gap of L_G),
    kap  := || Q_L [A, H] P_L ||_F                (Laplacian non-commutativity),
    delta:= || P_L (H^2 - v^2 A) P_L ||_F         (microscopic cone defect).

Assume g_L > 0. Then

  (i)   [infrared decoupling]   || Q_L H P_L ||_F <= kap / g_L,
        and the bound is attained.

  (ii)  [chirality descends]    H_L := P_L H P_L is self-adjoint and
        satisfies {Gamma, H_L} = 0, so the infrared block is chiral on a
        space of dimension 2 N_L.

  (iii) [cone inheritance]      H_L^2 = P_L H^2 P_L - (Q_L H P_L)^dagger
        (Q_L H P_L) exactly, hence

            || H_L^2 - v^2 P_L A P_L ||_F  <=  delta + (kap / g_L)^2 .

        The infrared block therefore satisfies the graph-Laplacian cone
        criterion with defect delta + (kap/g_L)^2.

  (iv)  [non-degeneracy]  If N_L = 1 then P_L A P_L = 0, the left-hand
        side of (iii) is independent of v, and the criterion determines no
        emergent velocity. A non-vacuous cone requires N_L >= 2.

Corollary (microscopic symmetry kills the leakage).
If G is a Cayley graph of an ABELIAN group Theta <= G_micro acting simply
transitively on V, and H is Theta-equivariant, then [A, H] = 0, so kap = 0
and (iii) holds with defect delta exactly. The abelian hypothesis is
necessary: Cay(S_3, {(12),(123),(132)}) is a 3-regular vertex-transitive
graph with a simply transitive NON-abelian automorphism group carrying an
equivariant graded H with [A, H] != 0.

WHAT THE PROPOSITION CONSUMES (the point of the exercise)
---------------------------------------------------------
* L_G   -- P_L, N_L, g_L and hence every hypothesis and every quantity in
           (i)-(iv) are functions of L_G. Deleting L_G does not make the
           proposition false; it makes it unstatable. This is certified
           operationally rather than rhetorically: `laplacian_is_not_
           determined_by_hamiltonian` exhibits ONE pair (Gamma, H) that is
           G-local for two different graphs (C_6 and K_6), so that every
           invariant of (Gamma, H) -- including the nullity bound -- takes
           identical values, while the infrared verdict is opposite.
           Hence no function of (Gamma, H) alone computes conclusion (iv).
* F     -- with dim F = 1 the grading is a scalar and {Gamma, H} = 0
           forces H = 0, so (ii) is empty. dim F >= 2 is used.
* Gamma -- carries conclusion (ii).
* H     -- carries (i) and (iii).
* G     -- enters through L_G and through the locality/degree data that
           make kap small. This module does NOT separate the role of G
           from the role of L_G; see SCOPE below.
* G_micro -- carries the Corollary; the abelian hypothesis is shown
           necessary.

RELATION TO STANDARD MATHEMATICS (stated plainly)
-------------------------------------------------
None of (i)-(iv) is new mathematics, and this module does not claim
otherwise. (i) is the standard off-diagonal decay estimate for a Sylvester
equation with separated spectra -- the elementary Hilbert-Schmidt case of
the Davis-Kahan sin-theta bound. (iii) is Schur-complement algebra. (ii)
is immediate once Gamma acts on the fiber and P_Lambda on the vertices.
(iv) is the observation that a one-dimensional spectral window is the
kernel.

The claim being made is not mathematical novelty; it is LOGICAL SHAPE. The
manuscript's existing candidate result, the nullity bound, is likewise
standard mathematics, and the objection to it was never that it is easy --
it was that it consumes only {Gamma, H}, so Definition 1 does no work in
its proof. What is offered here is a proposition of the form the objection
demands: its hypotheses and its conclusion are unstatable without L_G, its
Corollary is false without the commutativity of G_micro, conclusion (ii) is
false without Gamma, and every one of those dependencies is certified by an
explicit counterexample rather than asserted. On the same standard by which
Eq. (nullity) earns {Gamma, H}, this earns {G, F, Gamma, H, L_G, G_micro}.

SCOPE AND WHAT IS *NOT* ESTABLISHED
------------------------------------
0. The load-bearing content is (i), (iii), (iv) and the Corollary.
   Conclusion (ii) is bookkeeping: it records that the grading survives
   compression, and is included because it is what makes the infrared
   block "Dirac-like" rather than merely "small". It should not be
   advertised as a result.
1. The proposition is proved for the restricted class stated: finite
   connected simple graphs, fiber F = C^2, graded Hamiltonians of the form
   T (x) sigma_+ + h.c., grading Gamma = I (x) sigma_z. It is not proved
   for general fibers, general gradings, or infinite graphs.
2. The entry D is consumed only in its Hamiltonian presentation. Nothing
   here consumes the unitary-step presentation of D, and no claim is made
   that the tuple's D-as-unitary-step alternative is load-bearing.
3. G and L_G are not separated: every use of G in the proof is through
   L_G or through locality. The proposition therefore justifies the pair
   (G, L_G) jointly, not each independently.
4. (i) and (iii) are perturbative bounds, not existence theorems. They do
   not prove that an infrared Dirac block exists; they bound how badly the
   window fails to be invariant, and how the cone defect propagates from
   the microscopic operator to the compressed one, GIVEN that the
   microscopic defect delta is small. Producing a substrate with small
   delta remains an input, not an output.
5. The expander objects certify that N_L = 1 makes the criterion
   velocity-blind. They do NOT certify that every expander fails every
   geometric diagnostic, and they are not a falsification of the
   programme -- only a demonstration that the L_G-level hypothesis is not
   implied by the (Gamma, H)-level ones.
6. All certificates are finite explicit examples on at most 10 vertices
   (Hilbert space dimension at most 20). No asymptotic or family-level
   statement is machine-checked here; in particular "expander" is
   instantiated by two fixed bounded-degree graphs with an O(1) Laplacian
   gap (Petersen, and the prism Cay(S_3, .)), not by an infinite family.

Sustains:
- master_protospace.tex, Definition 1 (Sec. "The protospace datum"),
  and the minimality discussion that follows it.
"""
from __future__ import annotations

import itertools
from collections.abc import Iterable

import sympy as sp

from validators.graph_chiral_balance import characteristic_polynomial_is_even_or_odd
from validators.laplacian_projector import (
    cycle_laplacian,
    graph_laplacian,
    low_mode_count,
    low_mode_projector,
)

# --------------------------------------------------------------------------
# Fiber algebra and tensor conventions
# --------------------------------------------------------------------------

SIGMA_PLUS = sp.Matrix([[0, 1], [0, 0]])
SIGMA_MINUS = sp.Matrix([[0, 0], [1, 0]])
SIGMA_Z = sp.Matrix([[1, 0], [0, -1]])
FIBER_ID = sp.eye(2)


def kron(a: sp.Matrix, b: sp.Matrix) -> sp.Matrix:
    """Kronecker product with the (vertex, fiber) ordering used throughout."""
    return sp.Matrix(sp.kronecker_product(a, b))


def graded_hamiltonian(t_block: sp.Matrix) -> sp.Matrix:
    """H = T (x) sigma_+ + T^dagger (x) sigma_-, the graded Hamiltonian of P."""
    t_block = sp.Matrix(t_block)
    return kron(t_block, SIGMA_PLUS) + kron(t_block.T.conjugate(), SIGMA_MINUS)


def fiber_grading(n_vertices: int) -> sp.Matrix:
    """Gamma = I_V (x) sigma_z."""
    return kron(sp.eye(n_vertices), SIGMA_Z)


def laplacian_lift(lap: sp.Matrix) -> sp.Matrix:
    """A = L_G (x) I_F."""
    return kron(lap, FIBER_ID)


def frobenius_norm_squared(mat: sp.Matrix) -> sp.Expr:
    """||M||_F^2 = tr(M^dagger M), exact."""
    return sp.simplify((mat.H * mat).trace())


# --------------------------------------------------------------------------
# Spectral data of L_G: window dimension, infrared gap
# --------------------------------------------------------------------------

def _decide_le(lhs: sp.Expr, rhs: sp.Expr) -> bool:
    """Exact decision of lhs <= rhs for the algebraic numbers used here.

    Raises rather than guessing if sympy cannot decide the sign, so that no
    certificate in this module can silently degrade to a numerical one.
    """
    diff = sp.simplify(rhs - lhs)
    if diff.is_nonnegative is True:
        return True
    if diff.is_negative is True:
        return False
    raise ValueError(f"undecidable comparison: {lhs} <= {rhs}")


def laplacian_spectrum(lap: sp.Matrix) -> list[sp.Expr]:
    """Eigenvalues of L_G with multiplicity, as exact sympy expressions."""
    out: list[sp.Expr] = []
    for val, mult in lap.eigenvals().items():
        out.extend([sp.simplify(val)] * mult)
    return out


def _exact_min(values: Iterable[sp.Expr]) -> sp.Expr:
    """Minimum under the exact comparator `_decide_le` (no float fallback)."""
    values = list(values)
    best = values[0]
    for val in values[1:]:
        if _decide_le(val, best):
            best = val
    return best


def _exact_max(values: Iterable[sp.Expr]) -> sp.Expr:
    """Maximum under the exact comparator `_decide_le` (no float fallback)."""
    values = list(values)
    best = values[0]
    for val in values[1:]:
        if _decide_le(best, val):
            best = val
    return best


def window_dimension(lap: sp.Matrix, threshold) -> int:
    """N_Lambda = rank 1_{[0,Lambda]}(L_G)."""
    return low_mode_count(lap, threshold)


def window_projector(lap: sp.Matrix, threshold) -> sp.Matrix:
    """P_Lambda = 1_{[0,Lambda]}(L_G) (x) I_F, exact."""
    return kron(low_mode_projector(lap, threshold), FIBER_ID)


def infrared_gap(lap: sp.Matrix, threshold) -> sp.Expr:
    """g_Lambda = min{mu - lam : lam <= Lambda < mu, both in spec(L_G)}.

    Equals min(spec above Lambda) - max(spec at or below Lambda). Raises if
    the window is the whole spectrum (no gap is then defined).
    """
    spec = laplacian_spectrum(lap)
    thr = sp.nsimplify(threshold)
    low = [val for val in spec if _decide_le(val, thr)]
    high = [val for val in spec if not _decide_le(val, thr)]
    if not low or not high:
        raise ValueError("threshold does not split the spectrum")
    return sp.simplify(_exact_min(high) - _exact_max(low))


# --------------------------------------------------------------------------
# Exact comparison sqrt(a) <= sqrt(b) + c for nonnegative rationals
# --------------------------------------------------------------------------

def _sqrt_le_sqrt_plus(a: sp.Expr, b: sp.Expr, c: sp.Expr) -> bool:
    """Decide sqrt(a) <= sqrt(b) + c exactly, for rational a, b, c >= 0.

    sqrt(a) <= sqrt(b) + c  <=>  a - b - c^2 <= 2 c sqrt(b), and since both
    sides of the squared comparison are rational the decision is exact:
    true if a - b - c^2 <= 0, otherwise (a - b - c^2)^2 <= 4 c^2 b.
    """
    a, b, c = sp.nsimplify(a), sp.nsimplify(b), sp.nsimplify(c)
    residual = sp.simplify(a - b - c**2)
    if bool(residual <= 0):
        return True
    return bool(sp.simplify(residual**2 - 4 * c**2 * b) <= 0)


# --------------------------------------------------------------------------
# Graph library
# --------------------------------------------------------------------------

def cycle_hopping(n: int) -> sp.Matrix:
    """T = S - I on C_n: the nearest-neighbour forward difference."""
    shift = sp.Matrix(n, n, lambda i, j: 1 if (j - i) % n == 1 else 0)
    return shift - sp.eye(n)


def complete_graph_laplacian(n: int) -> sp.Matrix:
    """L(K_n) = n I - J, the extremal expander on n vertices."""
    return graph_laplacian(n, list(itertools.combinations(range(n), 2)))


PETERSEN_EDGES: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
    (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
]


def petersen_laplacian() -> sp.Matrix:
    """L of the Petersen graph: 3-regular, n = 10, spec {0, 2^(5), 5^(4)}.

    Chosen as the expander counterexample precisely because it is
    BOUNDED-DEGREE and local, so it cannot be dismissed as an artefact of
    the complete graph's unbounded degree.
    """
    return graph_laplacian(10, PETERSEN_EDGES)


def _s3_elements() -> list[tuple[int, ...]]:
    return list(itertools.permutations(range(3)))


def _s3_compose(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(a[b[i]] for i in range(3))


def s3_right_regular(t: tuple[int, ...]) -> sp.Matrix:
    """Right regular representation R(t): e_x -> e_{x t}."""
    els = _s3_elements()
    return sp.Matrix(6, 6, lambda i, j: 1 if els[i] == _s3_compose(els[j], t) else 0)


def s3_left_regular(t: tuple[int, ...]) -> sp.Matrix:
    """Left regular representation Lam(t): e_x -> e_{t x} (graph automorphisms)."""
    els = _s3_elements()
    return sp.Matrix(6, 6, lambda i, j: 1 if els[i] == _s3_compose(t, els[j]) else 0)


S3_TRANSPOSITION = (1, 0, 2)
S3_CYCLE = (1, 2, 0)
S3_CYCLE_INVERSE = (2, 0, 1)
S3_GENERATORS = [S3_TRANSPOSITION, S3_CYCLE, S3_CYCLE_INVERSE]


def cayley_s3_laplacian() -> sp.Matrix:
    """L of Cay(S_3, {(12),(123),(132)}): 3-regular, vertex-transitive, n = 6.

    The generating set is inverse-closed and identity-free (so the graph is
    simple and undirected) but is NOT a union of conjugacy classes, so the
    adjacency operator is not central in the group algebra. That is exactly
    what makes it a counterexample to the abelian hypothesis.
    """
    adj = sp.zeros(6, 6)
    for gen in S3_GENERATORS:
        adj += s3_right_regular(gen)
    return 3 * sp.eye(6) - adj


# --------------------------------------------------------------------------
# The three blocks of the Proposition
# --------------------------------------------------------------------------

def sylvester_identity_holds(lap: sp.Matrix, t_block: sp.Matrix, threshold) -> bool:
    """The mechanism behind (i): Q [A,H] P = A (Q H P) - (Q H P) A, exactly.

    P and Q commute with A because P is a spectral projector of L_G; this
    is the step that fails for a subspace which is not spectral for L_G,
    and it is what converts a commutator bound into a gap-divided bound.
    """
    n = lap.shape[0]
    a_op = laplacian_lift(lap)
    h_op = graded_hamiltonian(t_block)
    p_op = window_projector(lap, threshold)
    q_op = sp.eye(2 * n) - p_op
    x_op = sp.expand(q_op * h_op * p_op)
    lhs = sp.expand(q_op * (a_op * h_op - h_op * a_op) * p_op)
    rhs = sp.expand(a_op * x_op - x_op * a_op)
    return sp.simplify(lhs - rhs) == sp.zeros(2 * n, 2 * n)


def leakage_and_commutator(
    lap: sp.Matrix, t_block: sp.Matrix, threshold
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return (||Q H P||_F^2, ||Q [A,H] P||_F^2, g_Lambda), all exact."""
    n = lap.shape[0]
    a_op = laplacian_lift(lap)
    h_op = graded_hamiltonian(t_block)
    p_op = window_projector(lap, threshold)
    q_op = sp.eye(2 * n) - p_op
    x_op = sp.expand(q_op * h_op * p_op)
    comm = sp.expand(q_op * (a_op * h_op - h_op * a_op) * p_op)
    return (
        frobenius_norm_squared(x_op),
        frobenius_norm_squared(comm),
        infrared_gap(lap, threshold),
    )


def infrared_decoupling_bound_holds(
    lap: sp.Matrix, t_block: sp.Matrix, threshold
) -> bool:
    """Conclusion (i): g_Lambda^2 ||Q H P||_F^2 <= ||Q [A,H] P||_F^2.

    Stated in squared form so the comparison is a rational inequality.
    """
    leak_sq, comm_sq, gap = leakage_and_commutator(lap, t_block, threshold)
    return bool(sp.simplify(gap**2 * leak_sq - comm_sq) <= 0)


def chirality_descends_to_window(
    lap: sp.Matrix, t_block: sp.Matrix, threshold
) -> bool:
    """Conclusion (ii): H_Lambda = P H P is self-adjoint and {Gamma, H_L} = 0.

    Gamma acts on the fiber and P acts on the vertices, so they commute;
    the compression therefore inherits the grading. The check is exact and
    also confirms self-adjointness of the compression.
    """
    n = lap.shape[0]
    h_op = graded_hamiltonian(t_block)
    p_op = window_projector(lap, threshold)
    h_win = sp.expand(p_op * h_op * p_op)
    gamma = fiber_grading(n)
    zero = sp.zeros(2 * n, 2 * n)
    self_adjoint = sp.simplify(h_win - h_win.H) == zero
    anticommutes = sp.simplify(gamma * h_win + h_win * gamma) == zero
    return self_adjoint and anticommutes


def compression_identity_holds(
    lap: sp.Matrix, t_block: sp.Matrix, threshold
) -> bool:
    """The exact identity in (iii): P H^2 P - (P H P)^2 = (Q H P)^dagger (Q H P).

    This is projector algebra and holds for any orthogonal projector; it is
    the bridge that turns the leakage bound (i) into the cone bound (iii).
    It is recorded separately so that the L_G-dependent content of (iii) is
    not confused with this purely algebraic step.
    """
    n = lap.shape[0]
    h_op = graded_hamiltonian(t_block)
    p_op = window_projector(lap, threshold)
    q_op = sp.eye(2 * n) - p_op
    h_win = sp.expand(p_op * h_op * p_op)
    x_op = sp.expand(q_op * h_op * p_op)
    residual = sp.expand(p_op * h_op * h_op * p_op - h_win * h_win - x_op.H * x_op)
    return sp.simplify(residual) == sp.zeros(2 * n, 2 * n)


def cone_defects(
    lap: sp.Matrix, t_block: sp.Matrix, threshold, velocity=1
) -> tuple[sp.Expr, sp.Expr]:
    """Return (delta^2, delta_IR^2): microscopic and compressed cone defects.

    delta^2    = || P (H^2 - v^2 A) P ||_F^2
    delta_IR^2 = || (P H P)^2 - v^2 P A P ||_F^2
    """
    n = lap.shape[0]
    vel = sp.nsimplify(velocity)
    a_op = laplacian_lift(lap)
    h_op = graded_hamiltonian(t_block)
    p_op = window_projector(lap, threshold)
    h_win = sp.expand(p_op * h_op * p_op)
    micro = sp.expand(p_op * (h_op * h_op - vel**2 * a_op) * p_op)
    infrared = sp.expand(h_win * h_win - vel**2 * p_op * a_op * p_op)
    return frobenius_norm_squared(micro), frobenius_norm_squared(infrared)


def cone_inheritance_bound_holds(
    lap: sp.Matrix, t_block: sp.Matrix, threshold, velocity=1
) -> bool:
    """Conclusion (iii): delta_IR <= delta + (kappa / g_Lambda)^2, exactly.

    The comparison sqrt(delta_IR^2) <= sqrt(delta^2) + kappa^2/g^2 is
    decided by `_sqrt_le_sqrt_plus`, which reduces it to two rational
    inequalities, so no floating-point tolerance is involved.
    """
    _leak_sq, comm_sq, gap = leakage_and_commutator(lap, t_block, threshold)
    micro_sq, infrared_sq = cone_defects(lap, t_block, threshold, velocity)
    return _sqrt_le_sqrt_plus(infrared_sq, micro_sq, sp.simplify(comm_sq / gap**2))


# --------------------------------------------------------------------------
# Certified instances of the Proposition
# --------------------------------------------------------------------------

def cycle_protospace_has_exact_infrared_cone(n: int = 6, threshold=1) -> bool:
    """The cycle protospace saturates the Proposition with every defect zero.

    For G = C_n and T = S - I, the Laplacian commutes with H (both are
    circulant), so kappa = 0, the window is exactly invariant, the
    microscopic cone identity H^2 = A holds, and the compressed cone defect
    is zero. All four statements are exact.
    """
    lap = cycle_laplacian(n)
    t_block = cycle_hopping(n)
    leak_sq, comm_sq, _gap = leakage_and_commutator(lap, t_block, threshold)
    micro_sq, infrared_sq = cone_defects(lap, t_block, threshold, velocity=1)
    return (
        sp.simplify(comm_sq) == 0
        and sp.simplify(leak_sq) == 0
        and sp.simplify(micro_sq) == 0
        and sp.simplify(infrared_sq) == 0
    )


def defected_cycle_hopping(n: int = 6, weight=sp.Rational(1, 2)) -> sp.Matrix:
    """C_n hopping with one weakened bond: breaks the cyclic symmetry.

    Still G-local (support unchanged) and still graded, so it is a
    legitimate protospace Hamiltonian; but it no longer commutes with L_G,
    which is what makes it a non-trivial instance of the Proposition.
    """
    t_block = cycle_hopping(n)
    t_block[0, 1] = sp.nsimplify(weight)
    return t_block


def defected_cycle_instance_is_nontrivial(n: int = 6, threshold=1) -> bool:
    """The defected cycle exercises the Proposition with all quantities > 0.

    Certifies that the bounds of (i) and (iii) are being tested on an
    instance where kappa, the leakage and both cone defects are strictly
    positive -- i.e. the inequalities are not vacuously true because every
    term vanishes, as happens for the clean cycle.
    """
    lap = cycle_laplacian(n)
    t_block = defected_cycle_hopping(n)
    leak_sq, comm_sq, _gap = leakage_and_commutator(lap, t_block, threshold)
    micro_sq, infrared_sq = cone_defects(lap, t_block, threshold, velocity=1)
    return all(
        sp.simplify(q) > 0 for q in (leak_sq, comm_sq, micro_sq, infrared_sq)
    )


def decoupling_bound_is_attained() -> bool:
    """(i) is sharp: equality on the two-vertex graph K_2 at Lambda = 0.

    spec L(K_2) = {0, 2}, so the window is the constant mode, every
    commutator matrix element crosses the single gap g = 2, and the
    Sylvester division is exact rather than an estimate. Both sides are
    computed independently and compared as exact rationals.
    """
    lap = graph_laplacian(2, [(0, 1)])
    t_block = sp.Matrix([[0, 1], [3, 0]])
    leak_sq, comm_sq, gap = leakage_and_commutator(lap, t_block, 0)
    return sp.simplify(leak_sq) > 0 and sp.simplify(gap**2 * leak_sq - comm_sq) == 0


def decoupling_bound_is_strict_on_defected_cycle(n: int = 6, threshold=1) -> bool:
    """(i) is a genuine inequality, not an identity: strict on C_6.

    Here the commutator has weight on eigenvalue pairs separated by more
    than the minimal gap, so the division by g_Lambda over-counts and the
    bound is strict. Together with `decoupling_bound_is_attained` this
    shows the constant 1/g_Lambda is exactly right: attained but not
    always attained.
    """
    lap = cycle_laplacian(n)
    t_block = defected_cycle_hopping(n)
    leak_sq, comm_sq, gap = leakage_and_commutator(lap, t_block, threshold)
    return bool(sp.simplify(gap**2 * leak_sq - comm_sq) < 0)


# --------------------------------------------------------------------------
# Corollary: G_micro (abelian equivariance) kills the leakage
# --------------------------------------------------------------------------

def abelian_equivariance_forces_zero_commutator(n: int = 6) -> bool:
    """Corollary, positive half, with SYMBOLIC hopping amplitudes.

    C_n = Cay(Z_n, {+-1}) and Z_n is abelian, so both L_G and any
    Z_n-equivariant T lie in the commutative algebra of circulants. The
    check uses a general circulant T = sum_k c_k S^k with independent
    symbolic c_k, hence certifies the whole equivariant family at once
    rather than one representative.
    """
    coeffs = sp.symbols(f"c0:{n}", real=True)
    shift = sp.Matrix(n, n, lambda i, j: 1 if (j - i) % n == 1 else 0)
    t_block = sp.zeros(n, n)
    for k in range(n):
        t_block += coeffs[k] * (shift**k)
    lap = cycle_laplacian(n)
    a_op = laplacian_lift(lap)
    h_op = graded_hamiltonian(t_block)
    return sp.expand(a_op * h_op - h_op * a_op) == sp.zeros(2 * n, 2 * n)


def nonabelian_equivariance_allows_nonzero_commutator() -> bool:
    """Corollary, sharpness half: 'abelian' cannot be weakened to 'transitive'.

    On Cay(S_3, {(12),(123),(132)}) the left translations act simply
    transitively by graph automorphisms; the graded H built from the right
    translation R((123)) is equivariant for all of them; yet
    [L_G (x) I, H] != 0. So vertex-transitivity of G_micro is not enough --
    commutativity is what the Corollary uses.

    Returns True when the graph is genuinely 3-regular and simple, the left
    translations really are automorphisms, H really is equivariant, and the
    commutator is nevertheless nonzero.
    """
    lap = cayley_s3_laplacian()
    adj = 3 * sp.eye(6) - lap
    simple_regular = all(adj[i, i] == 0 for i in range(6)) and all(
        sum(adj.row(i)) == 3 for i in range(6)
    ) and adj == adj.T
    automorphisms = all(
        sp.simplify(s3_left_regular(t) * lap - lap * s3_left_regular(t)) == sp.zeros(6, 6)
        for t in _s3_elements()
    )
    h_op = graded_hamiltonian(s3_right_regular(S3_CYCLE))
    equivariant = all(
        sp.simplify(
            kron(s3_left_regular(t), FIBER_ID) * h_op
            - h_op * kron(s3_left_regular(t), FIBER_ID)
        )
        == sp.zeros(12, 12)
        for t in _s3_elements()
    )
    a_op = laplacian_lift(lap)
    commutes = sp.simplify(a_op * h_op - h_op * a_op) == sp.zeros(12, 12)
    return simple_regular and automorphisms and equivariant and not commutes


# --------------------------------------------------------------------------
# Sharpness: the L_G entry is load-bearing
# --------------------------------------------------------------------------

def window_of_dimension_one_annihilates_the_laplacian(
    lap: sp.Matrix, threshold
) -> bool:
    """If N_Lambda = 1 then P_Lambda L_G = 0, hence P_Lambda L_G P_Lambda = 0.

    Since the threshold is nonnegative the window always contains ker L_G,
    which is nonzero because L_G annihilates constants. A window of
    dimension one is therefore exactly ker L_G, and the compressed
    Laplacian vanishes identically. This is conclusion (iv)'s mechanism,
    exact, and it needs no connectedness assumption: a disconnected graph
    has dim ker L_G >= 2 and so cannot have N_Lambda = 1 at all.
    """
    if window_dimension(lap, threshold) != 1:
        return False
    proj = low_mode_projector(lap, threshold)
    return sp.simplify(proj * lap) == sp.zeros(*lap.shape)


def expander_window_is_trivial_below_the_gap() -> bool:
    """Petersen: N_Lambda = 1 for every Lambda < 2, and P_Lambda L_G = 0.

    spec L(Petersen) = {0, 2^(5), 5^(4)}. The graph is 3-regular, so it is
    as local as any lattice; what disqualifies it is a property of L_G
    alone. Because N_Lambda depends only on L_G, this statement is
    quantified over ALL protospaces on this graph and all H: no choice of
    Hamiltonian can supply the low modes the Laplacian does not have.
    """
    lap = petersen_laplacian()
    spectrum_ok = lap.eigenvals() == {sp.Integer(0): 1, sp.Integer(2): 5, sp.Integer(5): 4}
    thresholds = [0, sp.Rational(1, 2), 1, sp.Rational(3, 2), sp.Rational(199, 100)]
    trivial = all(window_dimension(lap, thr) == 1 for thr in thresholds)
    return spectrum_ok and trivial and window_of_dimension_one_annihilates_the_laplacian(lap, 1)


def expander_cone_criterion_is_velocity_blind() -> bool:
    """Petersen: the in-window cone defect does not depend on v at all.

    With N_Lambda = 1 the compressed Laplacian vanishes, so
    P (H^2 - v^2 A) P = P H^2 P for EVERY H: the v-dependence drops out
    identically, and the criterion constrains no emergent velocity. That
    is conclusion (iv). The quantified version is carried by
    `window_of_dimension_one_annihilates_the_laplacian`; here one concrete
    local chiral H is exhibited with v as a free positive symbol, and the
    resulting constant is shown nonzero so that the substrate is not
    merely trivially compliant.
    """
    vel = sp.symbols("v", positive=True)
    lap = petersen_laplacian()
    t_block = 3 * sp.eye(10) - lap  # the adjacency matrix: a local chiral hopping
    micro_sq, _ = cone_defects(lap, t_block, 1, velocity=vel)
    micro_sq = sp.simplify(micro_sq)
    return (not micro_sq.has(vel)) and sp.simplify(micro_sq) != 0


def cycle_cone_criterion_determines_velocity(n: int = 6) -> bool:
    """The cycle at the same threshold: the defect DOES depend on v.

    For C_n the in-window defect is 4(v^2 - 1)^2 -- strictly positive except
    at v = 1, where it vanishes. The emergent velocity is therefore pinned
    by the criterion. This is the contrast partner of
    `expander_cone_criterion_is_velocity_blind`: same fiber, same grading,
    same threshold, both graphs bounded-degree and local; only L_G differs,
    and only the graph with genuine low modes determines a velocity.
    """
    vel = sp.symbols("v", positive=True)
    lap = cycle_laplacian(n)
    t_block = cycle_hopping(n)
    micro_sq, _ = cone_defects(lap, t_block, 1, velocity=vel)
    micro_sq = sp.simplify(micro_sq)
    depends = micro_sq.has(vel)
    vanishes_at_one = sp.simplify(micro_sq.subs(vel, 1)) == 0
    nonzero_elsewhere = sp.simplify(micro_sq.subs(vel, 2)) != 0
    return depends and vanishes_at_one and nonzero_elsewhere


def same_order_cayley_graphs_disagree_on_velocity() -> bool:
    """The tightest form of the contrast: two Cayley graphs on SIX vertices.

    C_6 = Cay(Z_6, {+-1}) and the prism Cay(S_3, {(12),(123),(132)}) both
    have n = 6, are vertex-transitive, connected and bounded-degree (2 and
    3), and are given the same fiber, the same grading, the same threshold
    Lambda = 1 and the analogous local hopping (the adjacency operator).
    Nothing about size, regularity or locality separates them. Yet

        C_6    : N_Lambda = 3, defect = 4(v^2-1)^2, minimised at v = 1;
        prism  : N_Lambda = 1, defect = 162, independent of v.

    So the emergent velocity exists in one case and is undefined in the
    other, and the only datum that distinguishes them is L_G.
    """
    vel = sp.symbols("v", positive=True)
    cyc = cycle_laplacian(6)
    prism = cayley_s3_laplacian()
    cyc_defect = sp.simplify(cone_defects(cyc, cycle_hopping(6), 1, velocity=vel)[0])
    prism_defect = sp.simplify(
        cone_defects(prism, 3 * sp.eye(6) - prism, 1, velocity=vel)[0]
    )
    return (
        window_dimension(cyc, 1) == 3
        and window_dimension(prism, 1) == 1
        and cyc_defect.has(vel)
        and not prism_defect.has(vel)
        and sp.simplify(prism_defect) != 0
    )


def laplacian_is_not_determined_by_hamiltonian(n: int = 6, threshold=1) -> bool:
    """The load-bearing certificate: one (Gamma, H), two L_G, opposite verdicts.

    T = S - I on C_n is supported on the edges of C_n, hence also on the
    edges of K_n (which contains them). So the SAME Hilbert space, the SAME
    grading Gamma and the SAME Hamiltonian H underlie two admissible
    protospaces, differing only in the graph and its Laplacian. Every
    invariant of (Gamma, H) alone -- spectrum, nullity, the bound
    dim ker H = |A| + |B| - 2 rank T -- is therefore literally identical.
    Yet at the same threshold:

      * with L(C_n): N_Lambda >= 3, P_Lambda L P_Lambda != 0, and the
        window carries a nonzero compressed Hamiltonian;
      * with L(K_n): N_Lambda = 1, P_Lambda L P_Lambda = 0, and the
        compressed Hamiltonian vanishes identically.

    Hence no function of (Gamma, H) computes the infrared verdict, and L_G
    is independent data that Definition 1 must carry. This is the precise
    sense in which the entry earns its place.

    ANTICIPATED OBJECTION, and the honest answer. One may reply that G
    "should" be the minimal graph supporting H, which would recover L_G
    from H after all. Two things are true about that reply. First,
    Definition 1 as written imposes no such minimality, so if the author
    wants it, it is an ADDITIONAL datum that Definition 1 must state --
    which concedes the point that the Laplacian is not free. Second, and
    decisively, the reply does not touch
    `same_order_cayley_graphs_disagree_on_velocity`, where each Hamiltonian
    IS minimally local on its own graph, both graphs have six vertices, and
    the verdicts still differ. The sharpness therefore survives the
    minimality convention.
    """
    t_block = cycle_hopping(n)
    h_op = graded_hamiltonian(t_block)
    zero = sp.zeros(2 * n, 2 * n)

    lap_cycle = cycle_laplacian(n)
    lap_complete = complete_graph_laplacian(n)

    proj_cycle = window_projector(lap_cycle, threshold)
    proj_complete = window_projector(lap_complete, threshold)

    cycle_rich = (
        window_dimension(lap_cycle, threshold) >= 3
        and sp.simplify(
            proj_cycle * laplacian_lift(lap_cycle) * proj_cycle
        ) != zero
        and sp.simplify(proj_cycle * h_op * proj_cycle) != zero
    )
    complete_trivial = (
        window_dimension(lap_complete, threshold) == 1
        and sp.simplify(
            proj_complete * laplacian_lift(lap_complete) * proj_complete
        ) == zero
        and sp.simplify(proj_complete * h_op * proj_complete) == zero
    )
    return cycle_rich and complete_trivial


def hamiltonian_level_invariants_agree(n: int = 6) -> bool:
    """Companion to the previous check: the (Gamma, H) data really is shared.

    Certifies that the single operator H used with both Laplacians is
    graded, has E <-> -E paired spectrum (checked through the parity of its
    characteristic polynomial, not merely through a vanishing trace), and
    has the nullity 2(n - rank T) predicted by rank-nullity -- so the
    nullity bound of the manuscript, which consumes only {Gamma, H},
    returns one and the same number for the two protospaces that the
    Laplacian separates. Nothing here is new; the point is precisely that
    this level of data is blind to the distinction.
    """
    t_block = cycle_hopping(n)
    h_op = graded_hamiltonian(t_block)
    gamma = fiber_grading(n)
    graded = sp.simplify(gamma * h_op + h_op * gamma) == sp.zeros(2 * n, 2 * n)
    nullity = 2 * n - h_op.rank()
    return (
        graded
        and characteristic_polynomial_is_even_or_odd(h_op)
        and nullity == 2 * (n - t_block.rank())
    )


def unit_fiber_admits_no_nonzero_graded_hamiltonian() -> bool:
    """The F entry: with dim F = 1 the grading forces H = 0.

    A one-dimensional fiber makes Gamma a scalar +-1, so {Gamma, H} = 0
    reads 2 Gamma H = 0 and hence H = 0 for a generic symbolic H. There is
    then no infrared block to speak of, and conclusion (ii) is empty.
    Certified symbolically on a two-vertex substrate.
    """
    entries = sp.symbols("h0:4", real=True)
    h_op = sp.Matrix(2, 2, lambda i, j: entries[2 * i + j])
    for sign in (1, -1):
        gamma = sign * sp.eye(2)
        solution = sp.solve(
            list(sp.expand(gamma * h_op + h_op * gamma)), list(entries), dict=True
        )
        if solution != [{e: 0 for e in entries}]:
            return False
    return True


# --------------------------------------------------------------------------
# Negative controls: genuine mutations
# --------------------------------------------------------------------------

def mutated_gap_bound_holds(n: int = 6, threshold=1) -> bool:
    """MUTATION: replace g_Lambda by the full spectral spread of L_G.

    The spread max spec - min spec is larger than the infrared gap, so it
    would give a stronger (false) bound. Returning False is the expected
    outcome and certifies that the specific quantity g_Lambda -- the gap AT
    THE THRESHOLD -- is what the Proposition needs, not any convenient
    spectral scale.
    """
    lap = cycle_laplacian(n)
    t_block = defected_cycle_hopping(n)
    leak_sq, comm_sq, _gap = leakage_and_commutator(lap, t_block, threshold)
    spectrum = laplacian_spectrum(lap)
    spread = sp.simplify(_exact_max(spectrum) - _exact_min(spectrum))
    return bool(sp.simplify(spread**2 * leak_sq - comm_sq) <= 0)


def mutated_coordinate_window_bound_holds(n: int = 6, threshold=1) -> bool:
    """MUTATION: keep the window DIMENSION but use a non-spectral subspace.

    P is replaced by the coordinate projector onto the first N_Lambda
    vertices, which has the same rank but is not a spectral projector of
    L_G. The Sylvester step then fails because P no longer commutes with A,
    and the bound of (i) is violated. Returning False certifies that the
    Proposition consumes the SPECTRAL structure of L_G and not merely a
    subspace of the right size.
    """
    lap = cycle_laplacian(n)
    t_block = defected_cycle_hopping(n)
    dim = window_dimension(lap, threshold)
    coord = sp.diag(*([1] * dim + [0] * (n - dim)))
    p_op = kron(coord, FIBER_ID)
    q_op = sp.eye(2 * n) - p_op
    a_op = laplacian_lift(lap)
    h_op = graded_hamiltonian(t_block)
    x_op = sp.expand(q_op * h_op * p_op)
    comm = sp.expand(q_op * (a_op * h_op - h_op * a_op) * p_op)
    gap = infrared_gap(lap, threshold)
    return bool(
        sp.simplify(gap**2 * frobenius_norm_squared(x_op) - frobenius_norm_squared(comm))
        <= 0
    )


def mutated_ungraded_hamiltonian_keeps_window_chirality(n: int = 6, threshold=1) -> bool:
    """MUTATION: add an on-site term that breaks {Gamma, H} = 0.

    The perturbed operator is still G-local and still self-adjoint, so it
    is a perfectly good graph Hamiltonian -- it simply is not graded.
    Returning False certifies that conclusion (ii) is carried by the Gamma
    entry of the tuple and does not follow from locality plus
    self-adjointness.
    """
    lap = cycle_laplacian(n)
    h_op = graded_hamiltonian(cycle_hopping(n))
    onsite = kron(sp.diag(*[sp.Rational(k + 1, 10) for k in range(n)]), FIBER_ID)
    ungraded = h_op + onsite
    p_op = window_projector(lap, threshold)
    h_win = sp.expand(p_op * ungraded * p_op)
    gamma = fiber_grading(n)
    return sp.simplify(gamma * h_win + h_win * gamma) == sp.zeros(2 * n, 2 * n)


def mutated_noncirculant_step_has_zero_commutator(n: int = 6) -> bool:
    """MUTATION: break the Z_n equivariance of the Corollary's hypothesis.

    The defected cycle hopping is still G-local and still graded, but is no
    longer Z_n-equivariant. Returning False certifies that the Corollary's
    conclusion kappa = 0 really is driven by the microscopic symmetry
    G_micro and is not an automatic consequence of locality or grading.
    """
    lap = cycle_laplacian(n)
    a_op = laplacian_lift(lap)
    h_op = graded_hamiltonian(defected_cycle_hopping(n))
    return sp.simplify(a_op * h_op - h_op * a_op) == sp.zeros(2 * n, 2 * n)


def wrong_expander_window_is_nontrivial() -> bool:
    """MUTATION: claim the Petersen window is nontrivial at a subgap threshold.

    Returning False certifies that `expander_window_is_trivial_below_the_gap`
    is not an artefact of how the window is counted: the same counting
    machinery, asked for N_Lambda >= 2 on the same graph and threshold,
    answers no.
    """
    lap = petersen_laplacian()
    return window_dimension(lap, 1) >= 2
