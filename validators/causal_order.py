"""
The causal order of the substrate, and what can be reconstructed from it.

This module carries the causal rung of the master programme. The other
causality modules (`causality.py`, `causalidad_continuo_vs_discreto.py`,
`red_causal.py`) certify velocity bounds and finite propagation. What is at
stake here is the converse direction: how much of the emergent geometry is
fixed by the causal ORDER that the substrate itself generates --- the
executable content of Hawking--King--McCarthy / Malament (order fixes the
conformal class) together with the causal-set completion (order plus volume
counting fixes the metric outright).

CONVENTIONS AND THE DEFINITION OF THE SUBSTRATE ORDER
-----------------------------------------------------
The substrate is a coined quantum walk on a ring of `n_sites` vertices,
basis index = 2*site + coin, one-step unitary U = S (C x I) with C a real
rotation by the coin angle theta and S the coin-conditioned shift (coin 0
moves left by `shift` sites, coin 1 moves right by `shift`).  Lattice units:
one time step = Delta t = 1, one site = a = 1, so speeds are read in
sites/step.

An EVENT is a pair (n, x): step index n, site x.  The causal order is
defined operationally from the unitary, with no reference to any metric:

    p = (n_p, x_p)  precedes  q = (n_q, x_q)
      iff  n_q > n_p  and  || P_{x_q} U^{n_q - n_p} P_{x_p} || > tol,

where P_x projects onto the two-dimensional coin space at site x.  In words:
q is in the actual amplitude support of the walk run from site x_p for
n_q - n_p steps, for some coin state.  Everything downstream --- the
partial-order check, the cone reconstruction, the interval counting, the
dimension estimate --- consumes THIS relation.  Nothing is defined from a
target slope.  The tolerance is a numerical-zero threshold, not a physical
one: `influence_support_tolerance_has_margin` certifies that the smallest
nonzero block norm actually encountered exceeds it by five orders of
magnitude, so the support sets are tolerance-independent.

WHAT COMES OUT (and what does not)
----------------------------------
  (1) Locality: after n steps the amplitude outside the graph ball of radius
      n*shift is exactly zero, and the ball edge is attained.
  (2) The relation above is a strict partial order for every coin angle
      theta in [0, pi/2).  It is NOT transitive at the degenerate flip-flop
      point theta = pi/2, where the walk does not spread at all.
  (3) On the full event lattice the order has exactly TWO connected
      components: the coined walk moves one site per step, so the order is
      bipartite (a checkerboard), and the naive Malament bracketing is
      meaningless across components.  The reconstruction must be run inside
      one component; the causal future of a single origin is one, and is the
      event set used throughout.
  (4) MALAMENT, ON THE SUBSTRATE.  Bracketing sup |dx|/dn over related pairs
      against inf |dx|/dn over unrelated future pairs, on the walk's own
      order, returns

          sup_related = shift   exactly,
          inf_unrelated = shift * (N+1)/(N-1),

      for an event set of temporal extent N.  The recovered slope is the
      walk's STRICT support speed, computed independently from the step
      operator, and the bracket closes on it as O(1/N).
  (5) The recovered slope is NOT the infrared group velocity cos(theta).
      The causal order is fixed by the MAXIMAL signal speed of the substrate,
      which is the ballistic edge of the support cone (speed `shift`), while
      the Dirac cone of the low-energy theory has speed cos(theta) < shift
      for theta > 0.  The two coincide exactly at the massless point.  This
      is the correct result and it is a genuine substrate statement: the
      order-theoretic reconstruction sees the lattice cone, and the emergent
      metric cone sits strictly inside it whenever the walk is massive.
  (6) VOLUME COUNTING, ON THE SUBSTRATE.  The Alexandrov interval of the
      walk's order between two events is order-isomorphic to a product of
      two chains; counting its events with the substrate's own event density
      (rho = 1/2 events per unit coordinate area, a lattice constant, not a
      fit) recovers the Lorentzian interval sqrt(N^2 - X^2) --- not the
      coordinate separation --- to O(1/N).
  (7) MASS MATTERS FOR THE VOLUME RUNG, NOT FOR THE CONE RUNG.  At theta = 0
      the walk is two decoupled chiral movers; its causal set is two chains
      plus an origin, every Alexandrov interval is totally ordered, and the
      counting rung has no content.  The cone rung still returns slope 1.
      So the conformal reconstruction is mass-independent and the volume
      completion requires theta > 0.
  (8) POSITION DEPENDENCE COMES FROM THE GRAPH, NOT FROM THE COIN.  A walk
      with a strongly position-dependent coin angle theta(x) (the standard
      route to curved-background quantum walks) has EXACTLY the same causal
      order as the uniform walk: the order does not see the mass.  A cone
      that varies with position requires grading the substrate's own
      geometry --- here, the physical positions X(x) of the vertices.  On
      such a graded chain the local cone c(X) = local vertex spacing is
      recovered from the order in local windows, and the vertex index is the
      tortoise coordinate of the emergent metric ds^2 = c(X)^2 dt^2 - dX^2.
  (9) Negative controls: a nonlocal (wormhole) shift permutation is still
      unitary but has no linear cone and inverts the bracket; the flip-flop
      walk yields no partial order and no bracket; a uniform embedding shows
      no cone grading; certifying against cos(theta) instead of the strict
      speed fails; using the wrong event density spoils the proper time;
      a total order and a 3+1 sprinkling are rejected by the dimension
      estimator.

SCOPE --- WHAT IS *NOT* ESTABLISHED HERE
----------------------------------------
The reconstruction recovers the substrate's own lattice cone in 1+1
dimensions.  It is not a rigidity theorem: Malament's content is that the
chronological order determines the conformal metric among a priori
inequivalent geometries, and nothing here establishes that.  The regular
lattice is not a Lorentz-invariant sprinkling, so the causal set built here
is not boost-invariant and the Myrheim--Meyer value 1/2 is recovered with a
+1/(a+1) finite-size correction rather than as a sprinkling expectation.
The position-dependent cone of item (8) is supplied by the vertex embedding
and recovered from the order; the embedding is input, not derived from a
curved substrate dynamics.  The two `continuum_sprinkling_*` functions below
do NOT run on the substrate and are labelled as such: they define the order
from the target slope and are retained only as algorithm-consistency checks
(and as the only place a 3+1 cone appears, the substrate here being 1+1).

Sustains:
- master_protospace.tex, Part IV (operational reconstruction of the emergent
  geometry) and master_addendum_causal_symmetry_numeric.tex,
  Sec. sec:causal-order / Sec. sec:rungs-summary
"""
from __future__ import annotations

import numpy as np

_SEED = 20260708

# Numerical-zero threshold for "this site is in the support".  Audited by
# `influence_support_tolerance_has_margin`: over every walk and every step
# count used in this module, the smallest NONZERO block norm exceeds
# 1e5 * _SUPP_TOL, so the support sets do not depend on the threshold.
_SUPP_TOL = 1e-12


# ---------------------------------------------------------------------------
# The substrate: a local coined quantum walk on a line
# ---------------------------------------------------------------------------

def _walk_unitary(
    n_sites: int,
    theta: float,
    shift: int = 1,
    thetas=None,
    wormhole: tuple | None = None,
) -> np.ndarray:
    """Coined walk U = S (C x I): coin rotation, then coin-conditioned shift
    (coin 0 -> left by `shift`, coin 1 -> right by `shift`), periodic
    boundary.  Basis ordering: index = 2*site + coin.

    `thetas`, if given, is a per-site coin angle (a graded-coin walk).
    `wormhole = (a, b)`, if given, transposes the images of sites a and b in
    the right-shift permutation; the result is still exactly unitary (the
    shift stays a permutation) but is no longer local.
    """
    dim = 2 * n_sites
    pi_l = [(x - shift) % n_sites for x in range(n_sites)]
    pi_r = [(x + shift) % n_sites for x in range(n_sites)]
    if wormhole is not None:
        a, b = wormhole
        pi_r[a], pi_r[b] = pi_r[b], pi_r[a]
    U = np.zeros((dim, dim))
    for x in range(n_sites):
        th = theta if thetas is None else float(thetas[x])
        C = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        for cin in range(2):
            # coin acts first: |x, cin> -> sum_cout C[cout, cin] |x, cout>
            # then shift: cout = 0 moves left, cout = 1 moves right
            U[2 * pi_l[x] + 0, 2 * x + cin] += C[0, cin]
            U[2 * pi_r[x] + 1, 2 * x + cin] += C[1, cin]
    return U


def _site_support(psi: np.ndarray, n_sites: int, tol: float = _SUPP_TOL) -> np.ndarray:
    """Boolean site support of a walk state."""
    prob = psi.reshape(n_sites, 2)
    return (np.abs(prob) > tol).any(axis=1)


def support_outside_ball_is_exactly_zero(n_steps: int = 7) -> bool:
    """Locality of the substrate step: after n steps from a point seed the
    amplitude on every site with |x - x0| > n is exactly zero (below 1e-13),
    for a generic (Hadamard-angle) coin. This is the strict support cone."""
    n_sites = 4 * n_steps + 5
    x0 = n_sites // 2
    U = _walk_unitary(n_sites, np.pi / 4)
    psi = np.zeros(2 * n_sites)
    psi[2 * x0] = 1.0
    for _ in range(n_steps):
        psi = U @ psi
    supp = _site_support(psi, n_sites, tol=1e-13)
    sites = np.arange(n_sites)
    outside = np.abs(sites - x0) > n_steps
    return not supp[outside].any()


def support_cone_edge_is_attained(n_steps: int = 7) -> bool:
    """The strict cone is sharp: the edge sites x0 +- n carry nonzero
    amplitude (the ballistic component of the coined walk), so the causal
    cone of the substrate has slope exactly 1 site/step."""
    n_sites = 4 * n_steps + 5
    x0 = n_sites // 2
    U = _walk_unitary(n_sites, np.pi / 4)
    psi = np.zeros(2 * n_sites)
    psi[2 * x0] = 1.0
    for _ in range(n_steps):
        psi = U @ psi
    prob = np.abs(psi.reshape(n_sites, 2)) ** 2
    site_prob = prob.sum(axis=1)
    return site_prob[x0 - n_steps] > 1e-12 and site_prob[x0 + n_steps] > 1e-12


# ---------------------------------------------------------------------------
# The substrate causal order, built from the unitary
# ---------------------------------------------------------------------------

def _influence_stack(U: np.ndarray, n_sites: int, n_steps: int):
    """Return (R, margin) where R[m][xp, xq] is True iff the 2x2 block of
    U^m connecting site xp to site xq is nonzero, i.e. iff some state
    localized on xp has amplitude on xq after m steps.  `margin` is the
    smallest nonzero block norm encountered, used to audit `_SUPP_TOL`."""
    stack = []
    smallest = np.inf
    M = np.eye(2 * n_sites)
    for m in range(n_steps + 1):
        nrm = np.abs(M.reshape(n_sites, 2, n_sites, 2)).max(axis=(1, 3)).T
        stack.append(nrm > _SUPP_TOL)
        if m >= 1:
            nz = nrm[nrm > _SUPP_TOL]
            if nz.size:
                smallest = min(smallest, float(nz.min()))
        M = U @ M
    return stack, smallest


_CACHE: dict = {}


def _substrate_cone(theta: float, n_steps: int, shift: int = 1, thetas_key=None):
    """Events of the causal future of a single origin, and the order matrix
    of the substrate relation restricted to them.  Returns
    (events, rel, x0, n_sites, margin) with events a list of (n, site)."""
    key = (round(float(theta), 12), int(n_steps), int(shift), thetas_key)
    if key in _CACHE:
        return _CACHE[key]
    n_sites = 2 * shift * n_steps + 7
    x0 = n_sites // 2
    U = _walk_unitary(n_sites, theta, shift=shift)
    stack, margin = _influence_stack(U, n_sites, n_steps)
    events = [
        (n, int(x))
        for n in range(n_steps + 1)
        for x in np.where(stack[n][x0])[0]
    ]
    rel = _order_matrix(stack, events)
    out = (events, rel, x0, n_sites, margin)
    _CACHE[key] = out
    return out


def _order_matrix(stack, events) -> np.ndarray:
    """Boolean order matrix rel[i, j] = (event_i precedes event_j) from the
    influence stack.  Vectorized over time separation."""
    ns = np.array([e[0] for e in events])
    xs = np.array([e[1] for e in events])
    n = len(events)
    rel = np.zeros((n, n), dtype=bool)
    dn = ns[None, :] - ns[:, None]
    for m in range(1, int(dn.max()) + 1 if n else 1):
        ii, jj = np.where(dn == m)
        if ii.size:
            rel[ii, jj] = stack[m][xs[ii], xs[jj]]
    return rel


def _bracket(events, rel, coord) -> tuple[float, float]:
    """Malament/HKM bracketing on a given order: return
    (sup over related pairs of |dX|/dn, inf over unrelated future pairs).
    `coord` maps a site index to its physical position."""
    ns = np.array([e[0] for e in events], dtype=float)
    xs = np.array([coord(e[1]) for e in events], dtype=float)
    dn = ns[None, :] - ns[:, None]
    dx = np.abs(xs[None, :] - xs[:, None])
    future = dn > 0
    ratio = np.where(future, dx / np.where(future, dn, 1.0), np.nan)
    sup_related = np.nanmax(np.where(rel, ratio, np.nan))
    inf_unrelated = np.nanmin(np.where(future & ~rel, ratio, np.nan))
    return float(sup_related), float(inf_unrelated)


def _strict_support_speed(theta: float, n_steps: int, shift: int = 1) -> float:
    """The walk's own maximal signal speed, computed from the step operator
    alone: max over m of (support radius after m steps)/m.  This is the
    quantity the reconstruction must reproduce; it is never hardcoded."""
    n_sites = 2 * shift * n_steps + 7
    x0 = n_sites // 2
    U = _walk_unitary(n_sites, theta, shift=shift)
    stack, _margin = _influence_stack(U, n_sites, n_steps)
    speeds = []
    for m in range(1, n_steps + 1):
        reach = np.where(stack[m][x0])[0]
        speeds.append(np.abs(reach - x0).max() / m)
    return float(max(speeds))


def influence_support_tolerance_has_margin() -> bool:
    """The numerical-zero threshold `_SUPP_TOL = 1e-12` used to define the
    substrate order is auditable, not tuned: over every (coin angle, step
    count) pair used by this module, the SMALLEST nonzero 2x2 block norm of
    U^m is larger than 1e-7, i.e. five orders of magnitude above the
    threshold.  Support sets are therefore threshold-independent, and the
    order is not an artefact of the cutoff."""
    worst = np.inf
    for theta, n_steps, shift in (
        (0.2, 20, 1),
        (np.pi / 4, 40, 1),
        (1.1, 20, 1),
        (0.0, 20, 1),
        (np.pi / 2, 16, 1),
        (np.pi / 4, 14, 2),
        (np.pi / 4, 14, 3),
    ):
        _ev, _rel, _x0, _L, margin = _substrate_cone(theta, n_steps, shift=shift)
        worst = min(worst, margin)
    return worst > 1e5 * _SUPP_TOL


def substrate_order_is_strict_partial_order() -> bool:
    """The relation built from the walk unitary --- q is in the amplitude
    support of the walk run from x_p for n_q - n_p steps --- is a strict
    partial order (irreflexive, antisymmetric, transitive), checked
    exhaustively on the full rectangular event grid for several coin angles.

    Transitivity is NOT automatic here, and it is not universal.  Composing
    supports can fail when an intermediate site is missing from the support,
    so this is a genuine check on the dynamics and not a restatement of the
    definition.  It is verified for theta in {0.05, 0.2, pi/4, 1.1, 1.5},
    i.e. the whole massive range 0 < theta < pi/2.  It FAILS at the two
    endpoints, for opposite reasons, and both are certified separately:
    theta = 0 (`massless_support_relation_needs_transitive_closure`, where
    the support is only the two null edges) and theta = pi/2
    (`flipflop_walk_has_no_reconstructible_cone`, where the walk does not
    spread)."""
    n_steps, x_half = 6, 6
    for theta in (0.05, 0.2, np.pi / 4, 1.1, 1.5):
        n_sites = 4 * n_steps + 4 * x_half + 5
        x0 = n_sites // 2
        U = _walk_unitary(n_sites, theta)
        stack, _m = _influence_stack(U, n_sites, n_steps)
        events = [
            (n, x0 + dx)
            for n in range(n_steps + 1)
            for dx in range(-x_half, x_half + 1)
        ]
        rel = _order_matrix(stack, events)
        if rel.diagonal().any():
            return False
        if (rel & rel.T).any():
            return False
        composed = (rel.astype(int) @ rel.astype(int)) > 0
        if (composed & ~rel).any():
            return False
    return True


def massless_support_relation_needs_transitive_closure() -> bool:
    """The massless endpoint, stated exactly.  At theta = 0 the coin does not
    mix and the support after m steps is only {x0 - m, x0 + m}, so the raw
    influence relation contains only NULL-separated pairs.  On the full event
    lattice that relation is irreflexive and antisymmetric but NOT transitive:
    (0,0) influences (1,1) and (1,1) influences (2,0), yet (0,0) does not
    influence (2,0).

    What is certified here is the repair and its content: the transitive
    closure of the massless relation is EXACTLY EQUAL, edge for edge, to the
    raw relation of the massive walk, at theta = 0.2, pi/4 and 1.1 alike.  So
    the causal order of the substrate is one and the same object for every
    coin angle in [0, pi/2) once closure is taken --- the order is mass
    independent --- and the coin angle only controls whether the walk realizes
    that order directly or only through composition.  This is the order-level
    counterpart of `recovered_cone_is_the_strict_cone_not_the_infrared_cone`."""
    n_steps, x_half = 6, 6
    n_sites = 4 * n_steps + 4 * x_half + 5
    x0 = n_sites // 2
    events = [
        (n, x0 + dx)
        for n in range(n_steps + 1)
        for dx in range(-x_half, x_half + 1)
    ]

    def relation(theta):
        stack, _m = _influence_stack(_walk_unitary(n_sites, theta), n_sites, n_steps)
        return _order_matrix(stack, events)

    rel0 = relation(0.0)
    if rel0.diagonal().any() or (rel0 & rel0.T).any():
        return False
    composed = (rel0.astype(int) @ rel0.astype(int)) > 0
    if not (composed & ~rel0).any():
        return False  # transitivity must genuinely fail here

    closure = rel0.copy()
    for _ in range(2 * n_steps + 2):
        grown = closure | ((closure.astype(int) @ closure.astype(int)) > 0)
        if (grown == closure).all():
            break
        closure = grown
    else:
        return False

    for theta in (0.2, np.pi / 4, 1.1):
        if not (closure == relation(theta)).all():
            return False
    return True


def massless_order_is_transitive_on_a_single_causal_cone() -> bool:
    """Why the massless case may still be fed to the reconstruction.  Although
    the raw theta = 0 relation is not transitive on the full event lattice
    (previous function), it IS a strict partial order when restricted to the
    causal future of a single origin --- the event set every reconstruction in
    this module uses --- because there the only events are the two null rays
    and the tip, and composition along a ray stays on the ray.  Checked
    exhaustively at N = 20."""
    events, rel, _x0, _L, _m = _substrate_cone(0.0, 20)
    if rel.diagonal().any() or (rel & rel.T).any():
        return False
    composed = (rel.astype(int) @ rel.astype(int)) > 0
    return not bool((composed & ~rel).any())


def substrate_order_splits_into_two_parity_components() -> bool:
    """The coined walk moves exactly one site per step, so its causal order
    on the full event lattice is BIPARTITE: (n, x) and (n', x') are
    comparable only if n + x and n' + x' have the same parity.  The order
    therefore has exactly two connected components, and the naive Malament
    bracketing applied to the whole lattice is meaningless --- events of the
    opposite parity class sit at |dx|/dn arbitrarily small yet unrelated, so
    inf(unrelated) collapses below sup(related) and no bracket exists.

    This is why every reconstruction below runs inside ONE component (the
    causal future of a single origin).  Checked: exactly two components, and
    the whole-lattice bracket is inverted."""
    n_steps, x_half = 6, 6
    n_sites = 4 * n_steps + 4 * x_half + 5
    x0 = n_sites // 2
    U = _walk_unitary(n_sites, np.pi / 4)
    stack, _m = _influence_stack(U, n_sites, n_steps)
    events = [
        (n, x0 + dx)
        for n in range(n_steps + 1)
        for dx in range(-x_half, x_half + 1)
    ]
    rel = _order_matrix(stack, events)

    # comparability graph: count connected components
    undirected = rel | rel.T
    n = len(events)
    label = -np.ones(n, dtype=int)
    n_comp = 0
    for start in range(n):
        if label[start] >= 0:
            continue
        stack_dfs = [start]
        label[start] = n_comp
        while stack_dfs:
            a = stack_dfs.pop()
            for b in np.where(undirected[a])[0]:
                if label[b] < 0:
                    label[b] = n_comp
                    stack_dfs.append(b)
        n_comp += 1
    if n_comp != 2:
        return False

    # the two components are exactly the two parity classes
    parity = np.array([(n_ + x_) % 2 for (n_, x_) in events])
    same = (label == label[0]) == (parity == parity[0])
    if not same.all():
        return False

    # and the naive whole-lattice bracket is inverted
    sup_rel, inf_unrel = _bracket(events, rel, lambda x: x - x0)
    return bool(inf_unrel <= sup_rel)


# ---------------------------------------------------------------------------
# Malament / HKM, executed on the substrate's own order
# ---------------------------------------------------------------------------

def substrate_order_recovers_strict_cone_slope(n_steps: int = 20) -> bool:
    """MALAMENT ON THE SUBSTRATE.  Take the causal future of one origin under
    the walk's own influence relation, forget everything except that relation,
    and bracket

        sup over related pairs of |dx|/dn  <=  c  <  inf over unrelated
        future-pointing pairs of |dx|/dn.

    The recovered value is certified against the walk's STRICT SUPPORT SPEED
    computed independently by `_strict_support_speed` from the step operator
    --- no constant is hardcoded anywhere.  Because the lattice cone edge is
    attained, the lower end of the bracket is saturated exactly:
    sup_related = c, and inf_unrelated = c (N+1)/(N-1).

    Verified for coin angles theta in {0, 0.2, pi/4, 1.1}, i.e. massless and
    massive alike.

    SCOPE, stated plainly.  For a translation-invariant walk the equality
    sup_related = c is not deep: both sides are read off the same support
    sets, so that half of the check is a consistency requirement rather than
    a discovery.  The content that is NOT a corollary is threefold: that the
    relation is a genuine partial order at all (checked separately, and it
    fails at both endpoints of the coin range); that inf_unrelated lies
    strictly above c with the closed-form gap c (N+1)/(N-1), so the bracket
    really closes and does so as O(1/N); and that the value it closes on is
    the strict lattice cone and not the infrared cone cos(theta).  The
    strongest evidence that the pipeline reports the substrate rather than a
    constant is `reconstruction_tracks_substrate_shift_range`, where changing
    the walk changes the answer, and
    `mutated_wormhole_walk_breaks_substrate_reconstruction`, where the same
    pipeline fails on a nonlocal unitary.  This is emphatically NOT the
    Malament rigidity theorem, which asserts that inequivalent geometries
    cannot share a chronological order; nothing here establishes that."""
    for theta in (0.0, 0.2, np.pi / 4, 1.1):
        events, rel, x0, _L, _m = _substrate_cone(theta, n_steps)
        sup_rel, inf_unrel = _bracket(events, rel, lambda x: x - x0)
        c_walk = _strict_support_speed(theta, n_steps)
        if abs(sup_rel - c_walk) > 1e-12:
            return False
        if not (sup_rel <= c_walk < inf_unrel):
            return False
        predicted_inf = c_walk * (n_steps + 1) / (n_steps - 1)
        if abs(inf_unrel - predicted_inf) > 1e-12:
            return False
    return True


def substrate_cone_reconstruction_converges_with_lattice_size() -> bool:
    """Convergence of the substrate reconstruction.  The bracket
    [sup_related, inf_unrelated) always contains the walk's strict support
    speed c, and its width shrinks as

        inf_unrelated - sup_related  =  2 c / (N - 1),

    N being the temporal extent of the event set: the reconstruction is
    exact in the limit and first-order in the inverse lattice size.  Checked
    against the closed form at N = 8, 12, 20, 28, 40 (relative agreement
    better than 1e-12), and checked to be monotonically decreasing."""
    widths = []
    for n_steps in (8, 12, 20, 28, 40):
        events, rel, x0, _L, _m = _substrate_cone(np.pi / 4, n_steps)
        sup_rel, inf_unrel = _bracket(events, rel, lambda x: x - x0)
        c_walk = _strict_support_speed(np.pi / 4, n_steps)
        width = inf_unrel - sup_rel
        predicted = 2.0 * c_walk / (n_steps - 1)
        if abs(width - predicted) > 1e-12 * max(1.0, predicted):
            return False
        widths.append(width)
    return all(widths[i + 1] < widths[i] for i in range(len(widths) - 1))


def recovered_cone_is_the_strict_cone_not_the_infrared_cone() -> bool:
    """THE PHYSICS POINT.  The causal order is fixed by the substrate's
    MAXIMAL signal speed, so what the reconstruction returns is the strict
    lattice cone (speed 1 site/step, the ballistic edge of the support), not
    the infrared group velocity cos(theta) of the emergent Dirac cone.

    Checked: for theta in {0, 0.2, pi/4, 1.1} the recovered slope is 1 in
    every case, while cos(theta) takes the four distinct values
    {1, 0.980, 0.707, 0.454}; the recovered slope is therefore mass
    independent, and it equals cos(theta) exactly at, and only at, the
    massless point theta = 0.  The emergent metric cone is a strict subcone
    of the causal cone whenever the walk is massive."""
    recovered = []
    for theta in (0.0, 0.2, np.pi / 4, 1.1):
        events, rel, x0, _L, _m = _substrate_cone(theta, 20)
        sup_rel, _inf = _bracket(events, rel, lambda x: x - x0)
        recovered.append(sup_rel)
    # mass independence of the recovered cone
    if max(recovered) - min(recovered) > 1e-12:
        return False
    if abs(recovered[0] - 1.0) > 1e-12:
        return False
    # the IR cone genuinely differs, except at theta = 0
    ir = [np.cos(t) for t in (0.0, 0.2, np.pi / 4, 1.1)]
    if abs(ir[0] - recovered[0]) > 1e-15:
        return False
    if not all(v < recovered[0] - 1e-3 for v in ir[1:]):
        return False
    return True


def reconstruction_tracks_substrate_shift_range() -> bool:
    """The reconstruction is an OUTPUT of the dynamics, not a constant.
    Change the substrate --- give the shift a range of s sites per step,
    s = 1, 2, 3, which is still an exactly unitary local walk --- and the
    slope recovered from the order changes to exactly s, matching
    `_strict_support_speed` for each s, with the bracket
    [s, s (N+1)/(N-1)) scaling accordingly."""
    for shift in (1, 2, 3):
        n_steps = 14
        events, rel, x0, _L, _m = _substrate_cone(np.pi / 4, n_steps, shift=shift)
        sup_rel, inf_unrel = _bracket(events, rel, lambda x: x - x0)
        c_walk = _strict_support_speed(np.pi / 4, n_steps, shift=shift)
        if abs(c_walk - shift) > 1e-12:
            return False
        if abs(sup_rel - shift) > 1e-12:
            return False
        if not (sup_rel <= c_walk < inf_unrel):
            return False
        if abs(inf_unrel - shift * (n_steps + 1) / (n_steps - 1)) > 1e-12:
            return False
    return True


# ---------------------------------------------------------------------------
# Volume counting on the substrate order
# ---------------------------------------------------------------------------

def _interval_size(events, rel, p_index: int, q_index: int) -> int:
    """Cardinality of the open Alexandrov interval {r : p < r < q} in the
    substrate order."""
    return int((rel[p_index] & rel[:, q_index]).sum())


def substrate_counting_recovers_proper_time() -> bool:
    """CAUSAL-SET COMPLETION, ON THE SUBSTRATE.  Count the events of the
    walk's own Alexandrov interval I(p, q) and recover the proper time
    between its tips by the causal-set volume formula in 1+1,

        tau_est = sqrt( 2 N_I / (rho c) ),

    where c is the slope already recovered from the order --- recomputed here
    by `_strict_support_speed` from the step operator rather than written in
    as a literal, so it is 1 site/step because the walk says so --- and
    rho = 1/2 events per unit coordinate area is the substrate's own event
    density (one event per site per step, on one of the two parity classes ---
    a lattice constant, not a fit to the answer).

    Tips p = (0, 0) and q = (N, X = kappa N), so the target is the LORENTZIAN
    interval tau = sqrt(N^2 - X^2), not the coordinate time N.

    Convergence, with the constant predicted rather than fitted.  The
    substrate interval is the (u, v) rectangle a x b, a = (N+X)/2 + 1,
    b = (N-X)/2 + 1, minus its two tips, while tau^2/4 = (a-1)(b-1); so

        tau_est/tau = sqrt( (ab - 2) / ((a-1)(b-1)) )  =  1 + 1/N (1-kappa^2)
                                                          + O(N^-2),

    i.e. the relative error obeys  relerr * N * (1 - kappa^2) -> 2 from
    below.  Checked at N = 12, 20, 28, 36 for kappa = 0 and kappa = 1/2:
    relerr decreases monotonically in N, the scaled quantity increases
    monotonically toward the predicted 2, and it lies in [1.6, 2.0]
    throughout (it reaches 1.895 and 1.879 at N = 36; the raw errors there
    are 5.3% and 7.0%).  The residual is the one-lattice-unit boundary
    effect at each tip, and it is a discretization error that vanishes with
    the lattice, not a systematic bias."""
    tested = 0
    theta = np.pi / 4
    for kappa in (0.0, 0.5):
        errs, scaled = [], []
        for n_steps in (12, 20, 28, 36):
            X = int(round(kappa * n_steps))
            if (n_steps + X) % 2:
                continue
            events, rel, x0, _L, _m = _substrate_cone(theta, n_steps)
            index = {e: i for i, e in enumerate(events)}
            p, q = (0, x0), (n_steps, x0 + X)
            if p not in index or q not in index:
                return False
            n_int = _interval_size(events, rel, index[p], index[q])
            if n_int < 30:
                return False
            # Substrate event density (one event per two lattice cells, the
            # parity component) and the cone slope.  The slope is NOT a
            # literal: it is recomputed from the step operator by the same
            # helper the bracketing reconstruction is certified against, so a
            # change of shift range propagates here automatically.
            rho = 0.5
            c = _strict_support_speed(theta, n_steps)
            tau_est = np.sqrt(2.0 * n_int / (rho * c))
            tau_true = np.sqrt(float(n_steps) ** 2 - float(X) ** 2)
            rel_err = abs(tau_est - tau_true) / tau_true
            errs.append(rel_err)
            scaled.append(rel_err * n_steps * (1.0 - kappa ** 2))
            tested += 1
        if not all(errs[i + 1] < errs[i] for i in range(len(errs) - 1)):
            return False
        if not all(scaled[i + 1] > scaled[i] for i in range(len(scaled) - 1)):
            return False
        if not all(1.6 <= s <= 2.0 for s in scaled):
            return False
    return tested >= 8


def substrate_counting_tracks_lorentzian_not_coordinate_separation() -> bool:
    """The interval count knows the Lorentzian interval, not the coordinate
    time.  With tips at fixed N = 36 and spatial offsets X = 0, N/6, N/3,
    N/2, the counted volume gives tau_est whose ratios to tau_est(X = 0)
    follow sqrt(1 - (X/N)^2) --- measured deviations 0.0000, 0.0014, 0.0057,
    0.0140, accepted below 0.03 --- whereas the coordinate separation N is
    the same for all four.  A count that merely measured "how many events fit
    under q" could not do this: the interval count is sensitive to the
    boost-invariant separation, which is what the causal-set volume formula
    requires of it."""
    n_steps = 36
    events, rel, x0, _L, _m = _substrate_cone(np.pi / 4, n_steps)
    index = {e: i for i, e in enumerate(events)}
    taus, preds = [], []
    for X in (0, 6, 12, 18):
        q = (n_steps, x0 + X)
        if q not in index:
            return False
        n_int = _interval_size(events, rel, index[(0, x0)], index[q])
        taus.append(np.sqrt(2.0 * n_int / 0.5))
        preds.append(np.sqrt(1.0 - (X / n_steps) ** 2))
    ratios = [t / taus[0] for t in taus]
    if max(abs(r - p) for r, p in zip(ratios, preds)) > 0.03:
        return False
    # and the four tips are genuinely different: the largest offset is
    # clearly separated from the vertical one
    return ratios[-1] < 0.92


# ---------------------------------------------------------------------------
# Dimension from the substrate order
# ---------------------------------------------------------------------------

def _ordering_fraction(rel: np.ndarray) -> float:
    n = rel.shape[0]
    return 2.0 * float(rel.sum()) / (n * (n - 1))


def substrate_order_myrheim_meyer_dimension_is_two() -> bool:
    """Dimension from the substrate order alone.  Inside the walk's own
    Alexandrov interval between (0, 0) and (N, 0) the ordering fraction
    (related pairs / all pairs) approaches the Myrheim--Meyer value 1/2 for
    d = 2, with the exact finite-size form

        f(N) = (a + 3) / (2 (a + 1)),   a = N/2 + 1,

    i.e. f = 1/2 + 1/(a+1).  Two things are certified: f matches that closed
    form to 1e-12 at N = 16, 24, 32, 40 --- which says the substrate interval
    is order-isomorphic to a product of two chains of length a, the discrete
    signature of a 1+1 causal diamond --- and |f - 1/2| decreases
    monotonically, reaching 0.046 at N = 40.

    Honest caveats.  The excess above 1/2 is the lattice's null-related
    pairs: the cone edge is attained, so lightlike-separated events are
    causally related, which a Poisson sprinkling has measure zero of.  And
    the ordering fraction does not by itself certify manifoldlikeness: the
    massless walk's causal set (two chains) also returns ~1/2, for a wholly
    degenerate reason --- see
    `massless_substrate_causal_set_degenerates_to_two_chains`."""
    devs = []
    for n_steps in (16, 24, 32, 40):
        events, rel, x0, _L, _m = _substrate_cone(np.pi / 4, n_steps)
        index = {e: i for i, e in enumerate(events)}
        ip, iq = index[(0, x0)], index[(n_steps, x0)]
        inside = np.where(rel[ip] & rel[:, iq])[0]
        members = np.concatenate(([ip], inside, [iq]))
        sub = rel[np.ix_(members, members)]
        f = _ordering_fraction(sub)
        a = n_steps // 2 + 1
        closed = (a + 3.0) / (2.0 * (a + 1.0))
        if abs(f - closed) > 1e-12:
            return False
        devs.append(abs(f - 0.5))
    if not all(devs[i + 1] < devs[i] for i in range(len(devs) - 1)):
        return False
    return devs[-1] < 0.05


def massless_substrate_causal_set_degenerates_to_two_chains() -> bool:
    """MASS IS WHAT MAKES THE SUBSTRATE CAUSAL SET TWO-DIMENSIONAL.  At
    theta = 0 the coin does not mix, the walk is two decoupled chiral movers,
    and the support after n steps is only {x0 - n, x0 + n}.  Consequences,
    all checked here:

      - the causal future of the origin has 2N + 1 events, not the
        Theta(N^2) of the massive walk;
      - the event (N, 0), the timelike tip used by the volume rung, is not
        an event of the massless cone at all;
      - every Alexandrov interval is a CHAIN: the interval between the
        origin and (N, N) has ordering fraction exactly 1.

    So the conformal (cone) rung survives the massless limit --- slope 1 is
    still recovered --- but the volume-counting completion does not: at
    theta = 0 there are no timelike-separated tips and no 2D intervals to
    count.  The causal-set completion of the reconstruction requires a
    massive substrate."""
    n_steps = 20
    ev0, rel0, x0, _L, _m = _substrate_cone(0.0, n_steps)
    if len(ev0) != 2 * n_steps + 1:
        return False
    index = {e: i for i, e in enumerate(ev0)}
    if (n_steps, x0) in index:
        return False
    ip, iq = index[(0, x0)], index[(n_steps, x0 + n_steps)]
    inside = np.where(rel0[ip] & rel0[:, iq])[0]
    if len(inside) < 2:
        return False
    sub = rel0[np.ix_(inside, inside)]
    if abs(_ordering_fraction(sub) - 1.0) > 1e-12:
        return False
    # the massive walk at the same N is quadratically larger
    ev_m, _rel_m, _x0m, _Lm, _mm = _substrate_cone(np.pi / 4, n_steps)
    if len(ev_m) < 5 * len(ev0):
        return False
    # but the cone rung still works at theta = 0
    sup_rel, inf_unrel = _bracket(ev0, rel0, lambda x: x - x0)
    return bool(abs(sup_rel - 1.0) < 1e-12 and inf_unrel > 1.0)


# ---------------------------------------------------------------------------
# Position dependence: the graph carries it, the coin does not
# ---------------------------------------------------------------------------

def coin_grading_leaves_causal_order_flat() -> bool:
    """A position-dependent COIN does not curve the causal order.  Give the
    walk a strongly varying coin angle theta(x) ranging over [0.35, 0.85] ---
    the standard construction for quantum walks on a curved background, where
    a varying mass/coin produces a varying infrared metric --- and the
    influence relation is bit-for-bit IDENTICAL to that of the uniform walk,
    at every step separation.

    This is not a triviality: the support could have changed (it does at
    theta = 0 and theta = pi/2, where the coin degenerates).  What it shows
    is that the substrate causal order sees only the graph and the shift, so
    the position dependence of the emergent cone cannot come from the coin.
    It has to come from the geometry of the substrate itself --- see
    `graded_lattice_variable_cone_recovered_locally`.

    The grading is not a perturbation: the two step operators differ by
    0.2493 in SPECTRAL (operator) norm, certified below so that the number
    quoted in the manuscript is not free-floating.  Note the distinction ---
    the largest entrywise difference is 0.2217, a different quantity."""
    n_steps = 14
    n_sites = 2 * n_steps + 7
    thetas = 0.35 + 0.5 * np.sin(np.linspace(0.0, 3.0, n_sites))
    U_uniform = _walk_unitary(n_sites, 0.6)
    U_graded = _walk_unitary(n_sites, 0.0, thetas=thetas)
    if not np.allclose(U_graded.T @ U_graded, np.eye(2 * n_sites), atol=1e-12):
        return False
    if thetas.min() <= 0.0 or thetas.max() >= np.pi / 2:
        return False
    s_uniform, _m1 = _influence_stack(U_uniform, n_sites, n_steps)
    s_graded, _m2 = _influence_stack(U_graded, n_sites, n_steps)
    return all(
        bool((s_uniform[m] == s_graded[m]).all()) for m in range(n_steps + 1)
    )


def graded_coin_differs_from_uniform_in_operator_norm() -> bool:
    """The grading of `coin_grading_leaves_causal_order_flat` is a large
    deformation, not a perturbation: ||U_graded - U_uniform||_2 = 0.2493 to
    four figures.

    Without this, "the causal order does not see the coin" could be read as
    a statement about two walks that were nearly identical to begin with.
    The entrywise maximum is 0.2217 and is reported separately, because the
    two are easy to confuse and the manuscript quotes the operator norm."""
    n_steps = 14
    n_sites = 2 * n_steps + 7
    thetas = 0.35 + 0.5 * np.sin(np.linspace(0.0, 3.0, n_sites))
    diff = _walk_unitary(n_sites, 0.0, thetas=thetas) - _walk_unitary(n_sites, 0.6)
    spectral = float(np.linalg.norm(diff, 2))
    entrywise = float(np.abs(diff).max())
    return (
        abs(spectral - 0.2493) < 5e-5
        and abs(entrywise - 0.2217) < 5e-5
        and spectral > entrywise          # the two are genuinely distinct
    )


def graded_lattice_variable_cone_recovered_locally(eps: float = 0.03) -> bool:
    """A POSITION-DEPENDENT CONE, RECOVERED LOCALLY FROM THE WALK'S ORDER.
    The graded substrate is a chain whose vertices sit at non-uniform
    physical positions X(j) = (exp(eps j) - 1)/eps, so the physical spacing
    a(j) = X(j+1) - X(j) grows like exp(eps j).  The walk is unchanged --- one
    vertex per step --- so the ORDER is exactly the uniform substrate order
    of the functions above.  The emergent metric is
    ds^2 = c(X)^2 dt^2 - dX^2 with c(X) = a(X)/Delta t; since the walk's
    order reads |Delta j| <= Delta n in the vertex index j, and dX = c du
    turns that metric into c^2 (dt^2 - du^2), the vertex index IS the
    tortoise coordinate u of the emergent metric.  (That identification is an
    immediate consequence of the recovered relation, not a separate numerical
    check; what is checked numerically is the local slope below.)

    From the order alone, restricted to a window of 2w+1 vertices around a
    centre j*, the local cone slope is read off as the largest |dX|/dn over
    related pairs at the widest available baseline dn = 2w.  At eps = 0.03,
    w = 4 this returns c = 0.743 at j* = -10 and c = 1.353 at j* = +10,
    against the true local spacings 0.741 and 1.350 --- 0.2% --- and the two
    centres are cleanly distinguished (ratio 1.82).  The residual is the
    expected O((eps w)^2 / 6) window-averaging bias and grows with w exactly
    as predicted (1.0% at w = 8).

    The recovered RATIO c(+10)/c(-10) agrees with exp(20 eps) to machine
    precision, but that is not extra accuracy: an exponential grading is
    self-similar, so the window-averaging bias is a common factor that
    cancels in the ratio.  Only the absolute values carry the O((eps w)^2)
    error, and it is the absolute values that are checked at the 1% level.

    SCOPE: the position dependence is supplied by the vertex embedding, which
    is substrate data but is input here, not derived from a curved dynamics.
    What is certified is that the walk's own order plus the vertex positions
    determine the local cone, and that a uniform embedding yields no grading
    (see `wrong_uniform_embedding_shows_no_cone_grading`).  In particular
    this is not a reconstruction of a curved geometry from a curved
    substrate; that remains open."""
    n_steps = 24
    events, rel, x0, n_sites, _m = _substrate_cone(np.pi / 4, n_steps)
    j = np.arange(n_sites) - x0
    X = (np.exp(eps * j) - 1.0) / eps
    ns = np.array([e[0] for e in events])
    xs = np.array([e[1] for e in events])

    def local_slope(j_star: int, w: int) -> float:
        sel = np.where(np.abs((xs - x0) - j_star) <= w)[0]
        sub_rel = rel[np.ix_(sel, sel)]
        d_n = ns[sel][None, :] - ns[sel][:, None]
        d_X = np.abs(X[xs[sel]][None, :] - X[xs[sel]][:, None])
        ok = sub_rel & (d_n == 2 * w)
        if not ok.any():
            return float("nan")
        return float(d_X[ok].max() / (2 * w))

    w = 4
    ok_local = True
    for j_star in (-10, 10):
        c_est = local_slope(j_star, w)
        c_true = float(np.exp(eps * j_star))
        if not abs(c_est - c_true) / c_true < 0.01:
            ok_local = False
    if not ok_local:
        return False
    ratio_est = local_slope(10, w) / local_slope(-10, w)
    ratio_true = float(np.exp(eps * 20))
    if abs(ratio_est - ratio_true) / ratio_true > 1e-9:
        return False
    # the window bias grows with w, as the O((eps w)^2) estimate predicts
    err4 = abs(local_slope(10, 4) - np.exp(eps * 10)) / np.exp(eps * 10)
    err8 = abs(local_slope(10, 8) - np.exp(eps * 10)) / np.exp(eps * 10)
    return bool(err8 > err4 and err8 < 0.03 and ratio_est > 1.5)


# ---------------------------------------------------------------------------
# The infrared cone of the walk, from the step operator
# ---------------------------------------------------------------------------

def _bloch_step(k: float, theta: float) -> np.ndarray:
    """Bloch form of the one-step unitary at momentum k: U(k) = S(k) C with
    S(k) = diag(e^{+ik}, e^{-ik}) in the (coin 0 = left, coin 1 = right)
    convention of `_walk_unitary`."""
    C = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=complex,
    )
    S = np.diag([np.exp(1j * k), np.exp(-1j * k)])
    return S @ C


def bloch_form_matches_real_space_walk() -> bool:
    """The Bloch step operator used for the dispersion is the same operator
    as the real-space walk: applying `_walk_unitary` to the plane wave
    psi(x, c) = e^{ikx} v_c on a ring of L sites, at every allowed momentum
    k = 2 pi j / L and for both coin basis vectors, reproduces
    e^{ikx} (U(k) v)_c to 1e-13.  Without this the group-velocity statement
    below would be about a formula rather than about the substrate."""
    n_sites, theta = 12, 0.7
    U = _walk_unitary(n_sites, theta)
    x = np.arange(n_sites)
    for jj in range(n_sites):
        k = 2.0 * np.pi * jj / n_sites
        Uk = _bloch_step(k, theta)
        for c in range(2):
            v = np.zeros(2, dtype=complex)
            v[c] = 1.0
            psi = np.zeros(2 * n_sites, dtype=complex)
            psi[2 * x + c] = np.exp(1j * k * x)
            got = (U @ psi).reshape(n_sites, 2)
            want = np.exp(1j * k * x)[:, None] * (Uk @ v)[None, :]
            if np.max(np.abs(got - want)) > 1e-13:
                return False
    return True


def emergent_cone_inside_strict_cone() -> bool:
    """The infrared (metric) cone of the walk is bounded by the strict
    support cone, and the bound is saturated only at the massless point.

    The dispersion is taken from the EIGENVALUES of the Bloch step operator
    (certified against the real-space walk by `bloch_form_matches_real_space_walk`),
    not from a quoted formula: exp(-i omega(k)) are the eigenvalues of U(k),
    so cos(omega) = (1/2) tr U(k) = cos(k) cos(theta).  The group velocity
    v_g = d omega / d k then satisfies max_k |v_g| = cos(theta) <= 1, with
    equality exactly at theta = 0.  Checked on a fine k grid for several coin
    angles; the 5e-3 band is the finite-difference error of the numerical
    derivative near the band edge, not a physical tolerance."""
    ks = np.linspace(1e-6, np.pi - 1e-6, 6001)
    for theta in (0.0, 0.2, np.pi / 4, 1.1):
        # cos(omega) = (1/2) tr U(k), read off the Bloch step matrix itself
        cw = np.array(
            [0.5 * float(np.real(np.trace(_bloch_step(k, theta)))) for k in ks]
        )
        omega = np.arccos(np.clip(cw, -1.0, 1.0))
        vg = np.gradient(omega, ks)
        vmax = float(np.max(np.abs(vg)))
        if vmax > 1.0 + 1e-6:
            return False
        if abs(vmax - np.cos(theta)) > 5e-3:
            return False
    return bool(abs(np.cos(0.0) - 1.0) < 1e-15 and np.cos(0.2) < 1.0)


# ---------------------------------------------------------------------------
# Continuum-side consistency checks --- these do NOT run on the substrate
# ---------------------------------------------------------------------------

def _sprinkle_flat(n_pts: int, t_len: float, x_len: float, seed: int):
    rng = np.random.default_rng(seed)
    ts = rng.uniform(0.0, t_len, n_pts)
    xs = rng.uniform(-x_len / 2, x_len / 2, n_pts)
    return ts, xs


def continuum_sprinkling_recovers_cone_slope(c_true: float = 0.7) -> bool:
    """CONTINUUM-SIDE CHECK, NOT A SUBSTRATE RESULT.  On a Poisson sprinkling
    into 1+1 Minkowski, the order relation is DEFINED as |dx| < c_true dt,
    i.e. from the very slope that is then bracketed out of it.  This function
    therefore establishes only that the bracketing algorithm is correct and
    that a Lorentz-invariant sprinkling closes the bracket tightly (unlike
    the lattice, where sup_related saturates c exactly, here sup_related < c
    strictly and the gap is the sprinkling's discreteness).  It is retained
    as an algorithm control and as the continuum reference for
    `substrate_order_recovers_strict_cone_slope`, which is the substrate
    statement.  It is NOT the Malament rigidity theorem."""
    ts, xs = _sprinkle_flat(900, 1.0, 1.4, _SEED)
    dt = ts[None, :] - ts[:, None]
    dx = np.abs(xs[None, :] - xs[:, None])
    rel = (dt > 0) & (dx < c_true * dt)
    future = dt > 1e-9
    ratio = np.where(future, dx / np.where(future, dt, 1.0), np.nan)
    sup_related = np.nanmax(np.where(rel, ratio, np.nan))
    inf_unrelated = np.nanmin(np.where(future & ~rel, ratio, np.nan))
    bracket_ok = sup_related < c_true < inf_unrelated
    tight = (inf_unrelated - sup_related) / c_true < 0.05
    return bool(bracket_ok and tight)


def continuum_sprinkling_recovers_cone_slope_3plus1(c_true: float = 0.7) -> bool:
    """CONTINUUM-SIDE CHECK, NOT A SUBSTRATE RESULT.  The substrate of this
    module is 1+1 dimensional, so the (3+1) statement can only be made in the
    continuum, and there the relation is again defined from c_true.  What it
    shows is that the same bracketing recovers a single ISOTROPIC maximal
    speed from full spatial separations |dx| in R^3 --- one SO(3)-invariant
    cone rather than one null direction per axis --- so the 1+1 substrate
    result is not an artefact of the conformal class being one-dimensional.
    Extending the substrate reconstruction itself to 3+1 is open."""
    rng = np.random.default_rng(_SEED + 7)
    n_pts = 2600
    ts = rng.uniform(0.0, 1.0, n_pts)
    xs = rng.uniform(-0.7, 0.7, (n_pts, 3))
    dt = ts[None, :] - ts[:, None]
    dr = np.linalg.norm(xs[None, :, :] - xs[:, None, :], axis=2)
    rel = (dt > 0) & (dr < c_true * dt)
    future = dt > 1e-9
    ratio = np.where(future, dr / np.where(future, dt, 1.0), np.nan)
    sup_related = np.nanmax(np.where(rel, ratio, np.nan))
    inf_unrelated = np.nanmin(np.where(future & ~rel, ratio, np.nan))
    bracket_ok = sup_related < c_true < inf_unrelated
    tight = (inf_unrelated - sup_related) / c_true < 0.08
    return bool(bracket_ok and tight)


# ---------------------------------------------------------------------------
# Negative controls and genuine mutations
# ---------------------------------------------------------------------------

def flipflop_walk_has_no_reconstructible_cone() -> bool:
    """Degenerate substrate control (positive test of a negative fact).  At
    theta = pi/2 the coin is a pure flip: the walk oscillates between two
    sites and never spreads.  Then (i) the influence relation is NOT
    transitive, so it is not a causal order at all, and (ii) the bracket is
    inverted, inf_unrelated < sup_related, so no cone can be reconstructed.
    The reconstruction refuses to return a slope exactly when the substrate
    has no linearly growing support cone --- it does not manufacture one."""
    n_steps = 16
    events, rel, x0, _L, _m = _substrate_cone(np.pi / 2, n_steps)
    composed = (rel.astype(int) @ rel.astype(int)) > 0
    not_transitive = bool((composed & ~rel).any())
    sup_rel, inf_unrel = _bracket(events, rel, lambda x: x - x0)
    no_bracket = bool(inf_unrel <= sup_rel)
    return not_transitive and no_bracket


def wrong_infrared_speed_fails_substrate_bracket() -> bool:
    """MUTATION.  Run the identical substrate reconstruction but certify the
    recovered slope against the infrared group velocity cos(theta) instead of
    the strict support speed.  Returns the check result, which must be False
    for every massive coin angle: the order does not know about cos(theta).
    This is the control that separates the real result from the plausible
    wrong one, and it is a genuine mutation of the certified quantity, not a
    corollary of the positive test."""
    ok = True
    for theta in (0.2, np.pi / 4, 1.1):
        events, rel, x0, _L, _m = _substrate_cone(theta, 20)
        sup_rel, inf_unrel = _bracket(events, rel, lambda x: x - x0)
        c_ir = float(np.cos(theta))
        ok = ok and (sup_rel <= c_ir < inf_unrel)
    return ok


def _reconstruction_claim(U: np.ndarray, n_sites: int, n_steps: int) -> bool:
    """The full substrate reconstruction claim, applied to an arbitrary walk
    unitary: the support radius grows at a constant rate (a linear cone),
    and the Malament bracketing on the resulting order closes on that rate.
    True for the local coined walk; the mutations below feed it other
    unitaries."""
    x0 = n_sites // 2
    stack, _m = _influence_stack(U, n_sites, n_steps)
    events = [
        (n, int(x)) for n in range(n_steps + 1) for x in np.where(stack[n][x0])[0]
    ]
    rel = _order_matrix(stack, events)
    rates = [
        float(np.abs(np.where(stack[m][x0])[0] - x0).max()) / m
        for m in range(1, n_steps + 1)
    ]
    if max(rates) - min(rates) > 1e-12:
        return False
    c_walk = max(rates)
    sup_rel, inf_unrel = _bracket(events, rel, lambda x: x - x0)
    return bool(abs(sup_rel - c_walk) < 1e-12 and c_walk < inf_unrel)


def local_walk_passes_reconstruction_pipeline() -> bool:
    """Positive anchor for the mutation controls below: the unmodified local
    coined walk passes `_reconstruction_claim` --- linear cone growth plus a
    closing bracket --- on the same code path the mutations are fed
    through."""
    n_steps = 12
    n_sites = 4 * n_steps + 9
    return _reconstruction_claim(_walk_unitary(n_sites, np.pi / 4), n_sites, n_steps)


def mutated_wormhole_walk_breaks_substrate_reconstruction() -> bool:
    """MUTATION.  Replace the local shift permutation by one with a single
    transposition connecting two sites a third of the ring apart.  The step
    is still EXACTLY unitary and still a coined walk, but it is nonlocal.
    Feeds that unitary to the SAME `_reconstruction_claim` pipeline that the
    local walk passes, and returns its verdict, which must be False: the
    support radius jumps from 2 to 22 in one step, so there is no constant
    cone rate, and the bracket inverts (sup_related = 56 > inf_unrelated = 1).
    Locality of the step, not unitarity, is what makes the causal order
    geometric."""
    n_steps = 12
    n_sites = 4 * n_steps + 9
    x0 = n_sites // 2
    U = _walk_unitary(
        n_sites, np.pi / 4, wormhole=(x0 + 2, (x0 + 2 + n_sites // 3) % n_sites)
    )
    if not np.allclose(U.T @ U, np.eye(2 * n_sites), atol=1e-12):
        return True  # a non-unitary mutation would invalidate the control
    return _reconstruction_claim(U, n_sites, n_steps)


def wrong_uniform_embedding_shows_no_cone_grading() -> bool:
    """MUTATION.  Repeat `graded_lattice_variable_cone_recovered_locally`
    with the vertex embedding replaced by the uniform one X(j) = j, keeping
    the same walk and the same order.  Returns the graded claim's result,
    which must be False: both windows return c = 1 exactly and the ratio is
    1, so the local-cone estimator reports grading only when the substrate
    geometry actually has it."""
    eps = 0.03
    n_steps = 24
    events, rel, x0, n_sites, _m = _substrate_cone(np.pi / 4, n_steps)
    X = (np.arange(n_sites) - x0).astype(float)  # MUTATION: uniform spacing
    ns = np.array([e[0] for e in events])
    xs = np.array([e[1] for e in events])

    def local_slope(j_star: int, w: int) -> float:
        sel = np.where(np.abs((xs - x0) - j_star) <= w)[0]
        sub_rel = rel[np.ix_(sel, sel)]
        d_n = ns[sel][None, :] - ns[sel][:, None]
        d_X = np.abs(X[xs[sel]][None, :] - X[xs[sel]][:, None])
        ok = sub_rel & (d_n == 2 * w)
        if not ok.any():
            return float("nan")
        return float(d_X[ok].max() / (2 * w))

    w = 4
    for j_star in (-10, 10):
        c_est = local_slope(j_star, w)
        c_true = float(np.exp(eps * j_star))
        if not abs(c_est - c_true) / c_true < 0.01:
            return False
    return True


def mutated_event_density_fails_proper_time() -> bool:
    """MUTATION.  Repeat the substrate volume counting with the naive event
    density rho = 1 (one event per unit coordinate area) instead of the
    substrate's actual rho = 1/2 (the order is bipartite, so only half the
    lattice sites belong to a given causal component).  Returns the proper
    time check --- the identical acceptance band used by
    `substrate_counting_recovers_proper_time` --- which must be False: the
    estimator is then low by a factor sqrt(2), a 26% error whose scaled
    convergence quantity is 9.2 instead of the predicted ~2, and which no
    increase of the lattice removes.  The density entering the causal-set
    volume formula is a real substrate constant and the check is sensitive
    to it."""
    n_steps = 36
    events, rel, x0, _L, _m = _substrate_cone(np.pi / 4, n_steps)
    index = {e: i for i, e in enumerate(events)}
    n_int = _interval_size(events, rel, index[(0, x0)], index[(n_steps, x0)])
    rho_wrong = 1.0  # MUTATION: the true substrate density is 1/2
    tau_est = np.sqrt(2.0 * n_int / (rho_wrong * 1.0))
    rel_err = abs(tau_est - n_steps) / n_steps
    scaled = rel_err * n_steps  # kappa = 0
    return bool(1.6 <= scaled <= 2.0)


def dimension_estimator_rejects_total_order() -> bool:
    """Negative control: a chain (total order, dimension-1-like data) has
    ordering fraction 1, sharply distinct from the manifold value 1/2."""
    n = 300
    idx = np.arange(n)
    rel = idx[None, :] > idx[:, None]
    return abs(_ordering_fraction(rel) - 1.0) < 1e-12


def dimension_estimator_separates_3plus1() -> bool:
    """Negative control: a 3+1 Minkowski sprinkling has a markedly smaller
    ordering fraction than the 1+1 value 1/2 (analytically 1/20 = 0.05 in the
    diamond; we only require clear separation), so the estimator does not
    fake d = 2 on higher-dimensional data."""
    rng = np.random.default_rng(_SEED + 4)
    n_pts = 900
    ts = rng.uniform(0.0, 1.0, n_pts)
    ys = rng.uniform(-0.5, 0.5, (n_pts, 3))
    dt = ts[None, :] - ts[:, None]
    dr = np.linalg.norm(ys[None, :, :] - ys[:, None, :], axis=2)
    rel = (dt > 0) & (dr < dt)
    return _ordering_fraction(rel) < 0.3


def nonlocal_step_has_no_cone(n_sites: int = 40) -> bool:
    """Negative control: add a single wormhole edge (site 0 <-> N/2) to the
    one-step influence relation. Ball growth then violates the linear cone
    bound |ball(m)| <= 2m + 1, so no light cone --- and hence no Lorentzian
    metric --- can be reconstructed. Locality of the step is what makes the
    causal order geometric."""
    adj = np.zeros((n_sites, n_sites), dtype=bool)
    for x in range(n_sites):
        adj[x, (x - 1) % n_sites] = True
        adj[x, (x + 1) % n_sites] = True
    adj[0, n_sites // 2] = True
    adj[n_sites // 2, 0] = True
    reach = np.zeros(n_sites, dtype=bool)
    reach[0] = True
    violated = False
    for m in range(1, 6):
        reach = reach | (adj.astype(int).T @ reach.astype(int) > 0)
        if reach.sum() > 2 * m + 1:
            violated = True
            break
    # the local walk, by contrast, saturates the cone bound exactly
    adj_local = np.zeros((n_sites, n_sites), dtype=bool)
    for x in range(n_sites):
        adj_local[x, (x - 1) % n_sites] = True
        adj_local[x, (x + 1) % n_sites] = True
    reach_l = np.zeros(n_sites, dtype=bool)
    reach_l[0] = True
    local_ok = True
    for m in range(1, 6):
        reach_l = reach_l | (adj_local.astype(int).T @ reach_l.astype(int) > 0)
        if reach_l.sum() != 2 * m + 1:
            local_ok = False
            break
    return violated and local_ok
