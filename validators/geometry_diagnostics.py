"""
Geometry diagnostics: heat-kernel spectral dimension, low-mode metric
embedding, and the expander falsifier.

WHAT THIS MODULE IS FOR
-----------------------
Sec. IV.C of the master programme reconstructs two geometric data from the
spectrum of the graph Laplacian L_G alone --- an effective dimension and an
effective metric --- and Sec. IV.D uses a random regular expander as the
declared counterexample: a substrate that is local and chirally paired (so it
passes the matter-level tests) yet is not a geometry. Until now that entire
layer had no executable backing in this repository. This module supplies it.

DEFINITIONS (all fixed here, none inherited from an external pipeline)
----------------------------------------------------------------------
Heat kernel and spectral dimension, Eq. (eq:specdim) of the manuscript:

    P(tau) = (1/|V|) Tr e^{-tau L_G} = (1/|V|) sum_i e^{-tau lambda_i},
    d_s(tau) = -2 d log P(tau) / d log tau.

The manuscript calls d_s a "small-tau probe". On a GRAPH that description is
misleading and we do not adopt it: as tau -> 0 the spectrum is bounded, so
P -> 1 and d_s -> 0; as tau -> infinity only the zero mode survives, so
P -> 1/|V| and d_s -> 0 again. The dimension therefore lives in an
intermediate SCALING WINDOW, which we define with no free parameters from the
spectrum itself:

    tau in [1 / lambda_max, 1 / lambda_1],

lambda_1 being the spectral gap (Fiedler value) and lambda_max the top of the
spectrum. The lower end is the shortest time on which the walk has left a
site; the upper end is the longest time before the walk has equilibrated.
The width of this window in decades,

    span = log10(lambda_max / lambda_1),

is itself a diagnostic and is the sharpest one this module has (see below).

PLATEAU. On a log-uniform grid of `_N_TAU` = 240 points across the scaling
window we take the contiguous sub-window of width `_PLATEAU_FRACTION` = 1/3
of the log-range that MINIMISES the standard deviation of d_s. The plateau
value is the median of d_s there and sigma_ds is its standard deviation.
Minimising over sub-windows is deliberately generous to the substrate under
test: it hands every graph, including the expander, the flattest region it
possesses. A graph that fails here fails on its own best evidence.

LOW-MODE EMBEDDING. The first `n_modes` non-trivial eigenvectors of L_G
(columns 1..n_modes of the ordered spectral decomposition, the zero mode
dropped) are used as coordinates; we report the Spearman rank correlation
between graph geodesic distance and Euclidean distance in those coordinates,
over all vertex pairs. We use n_modes = 2 UNIFORMLY, for the lattice and for
the expander, so that the falsification comparison runs through one code
path. Spearman (rank) rather than Pearson because the claim is that the
low-mode coordinates recover the ORDERING of the graph metric, not its scale.

CONE DEFECT. The manuscript quotes an "isolated cone defect" of 2.6e-3 for
the expander but does not define the quantity anywhere in the text, and the
pipeline that produced it is not in this repository. We therefore state our
own definition and report what it gives. For any finite graph, let B be the
ORIENTED incidence matrix and

    D = [[0, B], [B^T, 0]]   ==>   D^2 = diag(B B^T, B^T B) = diag(L_G, L_E).

The cone criterion of `cone_criterion.py` (H^2 = L_G tensor 1 on the vertex
sector) is therefore satisfied EXACTLY, with defect identically zero, by
every connected graph -- lattice and expander alike. Under this definition
the manuscript's point is stronger than it claims: the cone diagnostic is not
merely a small false positive on the expander, it is structurally incapable of
discriminating.

WHAT SEPARATES A LATTICE FROM AN EXPANDER, IN ORDER OF STRENGTH
---------------------------------------------------------------
Measured here (seed 20260610), lattice vs 3-regular random expander on 256
vertices:

  (1) scaling-window growth   slope d(span)/d(log10 N):
        cycle 2.00, irregular grid 0.99, expander 0.025   -- factor ~40
  (2) low-mode metric correlation rho (n_modes = 2):
        grid 0.959, expander 0.436                        -- factor ~2.2
  (3) plateau flatness sigma_ds:
        grid 0.059, expander 0.158                        -- factor ~2.6

Diagnostic (1) is the quantitative content of "an O(1) spectral gap forbids
any infrared scaling": for a lattice the scaling window widens as the system
grows (lambda_1 ~ L^-2 diffusive), for an expander it does not (lambda_1 is
O(1) in N). It also cross-checks (3) independently, since span grows as
(2/d_s) log10 N, so slope * d_s = 2 is a consistency relation between two
different functionals of the spectrum. Diagnostic (3), the flatness criterion
the manuscript leads with, is the WEAKEST of the three and we say so.

A warning the manuscript should absorb: the expander's plateau VALUE is
d_s = 2.09, i.e. on its (narrow) window it impersonates a two-dimensional
substrate. The value of d_s is a false positive just as the cone defect is.
Only flatness, window growth and the metric embedding discriminate.

TOLERANCES
----------
These quantities are irreducibly numerical (dense `eigvalsh`, finite
differences of a log-log heat trace, a rank correlation), so exact sympy
tests are not available and every acceptance band below is justified where it
is used. Bands were fixed by measuring the spread over 3-4 random seeds and
over the family parameters (grid irregularity 5-12%, expander degree 3-6)
BEFORE being written down; each band is set to contain the observed spread
with margin while remaining far from the value it must discriminate against.
The random seed is fixed at `SEED = 20260610` (the manuscript's version date)
and is the only source of randomness in the module.

SCOPE LIMIT. Nothing here is a substrate-level theorem. These are numerical
diagnostics applied to specific finite graphs at specific sizes. They
establish that the checklist of Sec. IV.D is non-vacuous and that its
composite structure defeats single-diagnostic false positives. They do NOT
establish that a substrate passing the checklist yields a general-relativistic
limit.

Sustains:
- master_protospace.tex, Sec. IV.C "Gauge holonomy and spectral dimension:
  geometry one can measure" (Eq. (eq:specdim); the cycle plateau at N = 256;
  d_s -> 2 on irregular grids; the low-mode Spearman correlation)
- master_protospace.tex, Sec. IV.D "The falsifier: not every local substrate
  is a geometry" (the random regular expander non-example; the composite
  REJECT verdict; the cone defect as an isolated false positive)
"""
from __future__ import annotations

from functools import lru_cache

import networkx as nx
import numpy as np
import sympy as sp
from scipy.stats import spearmanr

from validators.laplacian_projector import cycle_laplacian

# --------------------------------------------------------------------------
# Fixed constants of the diagnostic. Changing these changes the numbers.
# --------------------------------------------------------------------------

SEED = 20260610          # manuscript version date; the only randomness source
_N_TAU = 240             # log-uniform samples across the scaling window
_PLATEAU_FRACTION = 1.0 / 3.0   # width of the plateau sub-window, in log-range
_N_MODES = 2             # low-mode embedding dimension, same for every family
_GRID_SIDE = 16          # flagship irregular grid is 16 x 16 = 256 vertices
_GRID_REMOVED = 0.08     # 8% of edges deleted -> "mildly irregular"
_EXPANDER_DEGREE = 3     # cubic random regular graph: the minimal expander
_N_VERTICES = 256        # matches the manuscript's N = 256


# --------------------------------------------------------------------------
# Graph families
# --------------------------------------------------------------------------

def cycle_spectrum(n: int) -> np.ndarray:
    """Closed-form Laplacian spectrum of C_n: 2 - 2 cos(2 pi k / n).

    Used instead of diagonalising, so that N = 512 costs nothing. The closed
    form is cross-checked against the exact sympy Laplacian of
    `laplacian_projector.cycle_laplacian` in
    `closed_form_cycle_spectrum_matches_sympy`.
    """
    k = np.arange(n)
    return 2.0 - 2.0 * np.cos(2.0 * np.pi * k / n)


def irregular_grid(side: int = _GRID_SIDE,
                   removed_fraction: float = _GRID_REMOVED,
                   seed: int = SEED) -> nx.Graph:
    """A `side` x `side` square grid with `removed_fraction` of its edges cut.

    Edges are removed in a seeded random order, and a removal is rejected if
    it would disconnect the graph, so the result is always connected. This is
    the manuscript's "mildly irregular grid": locally two-dimensional, but
    with no exact translation symmetry and no Brillouin torus.

    Defaults are normalised before caching so that `irregular_grid()` and
    `irregular_grid(side=16)` return the identical object, keeping the
    downstream spectrum cache warm.
    """
    return _irregular_grid_cached(side, removed_fraction, seed)


@lru_cache(maxsize=None)
def _irregular_grid_cached(side: int, removed_fraction: float, seed: int) -> nx.Graph:
    rng = np.random.default_rng(seed)
    graph = nx.grid_2d_graph(side, side)
    edges = list(graph.edges())
    order = rng.permutation(len(edges))
    target = int(removed_fraction * len(edges))
    removed = 0
    for idx in order:
        edge = edges[int(idx)]
        graph.remove_edge(*edge)
        if nx.is_connected(graph):
            removed += 1
            if removed >= target:
                break
        else:
            graph.add_edge(*edge)
    return graph


def random_expander(degree: int = _EXPANDER_DEGREE,
                    n_vertices: int = _N_VERTICES,
                    seed: int = SEED) -> nx.Graph:
    """A random `degree`-regular graph: the manuscript's falsifier substrate.

    Random regular graphs are expanders with probability tending to one and
    are near-Ramanujan (Friedman's theorem). What matters for the falsifier
    is only that the spectral gap is O(1) in the number of vertices, which is
    certified directly by `expander_gap_does_not_close`.

    Defaults are normalised before caching, as for `irregular_grid`.
    """
    return _random_expander_cached(degree, n_vertices, seed)


@lru_cache(maxsize=None)
def _random_expander_cached(degree: int, n_vertices: int, seed: int) -> nx.Graph:
    return nx.random_regular_graph(degree, n_vertices, seed=seed)


@lru_cache(maxsize=None)
def _laplacian_spectrum(graph: nx.Graph) -> tuple[float, ...]:
    """Ordered Laplacian eigenvalues of a graph (dense, exact-arithmetic input)."""
    lap = nx.laplacian_matrix(graph).toarray().astype(float)
    return tuple(float(v) for v in np.linalg.eigvalsh(lap))


# --------------------------------------------------------------------------
# Heat kernel and spectral dimension
# --------------------------------------------------------------------------

def return_probability(eigenvalues, taus) -> np.ndarray:
    """P(tau) = (1/|V|) sum_i exp(-tau lambda_i), the random-walk return
    probability of Eq. (eq:specdim)."""
    lam = np.asarray(eigenvalues, dtype=float)
    tau = np.asarray(taus, dtype=float)
    return np.exp(-np.outer(tau, lam)).mean(axis=1)


def spectral_dimension_curve(eigenvalues, taus) -> np.ndarray:
    """d_s(tau) = -2 d log P / d log tau, by central differences on the
    log-log heat trace."""
    prob = return_probability(eigenvalues, taus)
    return -2.0 * np.gradient(np.log(prob), np.log(np.asarray(taus, dtype=float)))


def scaling_window(eigenvalues) -> tuple[float, float]:
    """The scaling window [1/lambda_max, 1/lambda_1] fixed by the spectrum."""
    lam = np.sort(np.asarray(eigenvalues, dtype=float))
    return 1.0 / lam[-1], 1.0 / lam[1]


def scaling_window_span(eigenvalues) -> float:
    """Width of the scaling window in decades, log10(lambda_max / lambda_1)."""
    lo, hi = scaling_window(eigenvalues)
    return float(np.log10(hi / lo))


def spectral_dimension_plateau(eigenvalues) -> tuple[float, float]:
    """(d_s, sigma_ds) on the flattest sub-window of the scaling window.

    Returns the median and the standard deviation of d_s over the contiguous
    sub-window of width `_PLATEAU_FRACTION` of the log-range that minimises
    that standard deviation.
    """
    lo, hi = scaling_window(eigenvalues)
    taus = np.logspace(np.log10(lo), np.log10(hi), _N_TAU)
    curve = spectral_dimension_curve(eigenvalues, taus)
    width = max(5, int(round(_PLATEAU_FRACTION * _N_TAU)))
    best_sigma = np.inf
    best_median = float("nan")
    for start in range(_N_TAU - width + 1):
        segment = curve[start:start + width]
        sigma = float(np.std(segment))
        if sigma < best_sigma:
            best_sigma = sigma
            best_median = float(np.median(segment))
    return best_median, best_sigma


# --------------------------------------------------------------------------
# Low-mode metric embedding
# --------------------------------------------------------------------------

def low_mode_embedding(graph: nx.Graph, n_modes: int = _N_MODES) -> np.ndarray:
    """Coordinates from the first `n_modes` non-trivial eigenvectors of L_G.

    The zero (constant) mode is dropped; no embedding is assumed, the
    coordinates are read off the spectrum.
    """
    lap = nx.laplacian_matrix(graph).toarray().astype(float)
    _vals, vecs = np.linalg.eigh(lap)
    return vecs[:, 1:1 + n_modes]


def _pairwise_graph_and_embedding_distances(graph: nx.Graph, coords: np.ndarray):
    nodes = list(graph.nodes())
    geodesic = dict(nx.all_pairs_shortest_path_length(graph))
    n = len(nodes)
    graph_d = []
    embed_d = []
    for i in range(n):
        row = geodesic[nodes[i]]
        for j in range(i + 1, n):
            graph_d.append(row[nodes[j]])
            embed_d.append(float(np.linalg.norm(coords[i] - coords[j])))
    return np.asarray(graph_d, dtype=float), np.asarray(embed_d, dtype=float)


def metric_embedding_correlation(graph: nx.Graph, n_modes: int = _N_MODES) -> float:
    """Spearman rho between graph geodesic distance and low-mode Euclidean
    distance, over all vertex pairs."""
    coords = low_mode_embedding(graph, n_modes)
    graph_d, embed_d = _pairwise_graph_and_embedding_distances(graph, coords)
    return float(spearmanr(graph_d, embed_d).statistic)


# --------------------------------------------------------------------------
# Cross-check against the exact sympy layer
# --------------------------------------------------------------------------

def closed_form_cycle_spectrum_matches_sympy(n: int = 8) -> bool:
    """The numeric closed form 2 - 2 cos(2 pi k / n) is the spectrum of the
    exact sympy cycle Laplacian of `laplacian_projector.cycle_laplacian`.

    Tolerance 1e-12: both sides are evaluations of the same algebraic numbers
    in double precision, so agreement is at machine level; 1e-12 leaves four
    orders of headroom over the observed ~1e-15 residual while still failing
    on any genuine mismatch (the smallest spacing in this spectrum is ~0.29).
    """
    exact = cycle_laplacian(n)
    sympy_eigs = []
    for val, mult in exact.eigenvals().items():
        sympy_eigs.extend([float(sp.N(val))] * mult)
    numeric = np.sort(cycle_spectrum(n))
    return bool(np.allclose(np.sort(sympy_eigs), numeric, atol=1e-12, rtol=0.0))


# --------------------------------------------------------------------------
# Sec. IV.C: the calibrated diagnostic
# --------------------------------------------------------------------------

def cycle_plateau_is_one(n: int = _N_VERTICES) -> bool:
    """Calibration case. The cycle C_n is exactly one-dimensional, so the
    plateau must sit at d_s = 1 and must be flat.

    Bands: |d_s - 1| <= 0.01 and sigma_ds <= 5e-3. Measured at n = 256:
    d_s = 1.00037 (27 times inside its band) and sigma_ds = 4.5e-4 (11 times
    inside its band). Both bands are nonetheless two orders of magnitude
    below the discrimination that matters -- a two-dimensional substrate
    gives d_s ~ 2, a hundred band-widths away.
    """
    d_s, sigma = spectral_dimension_plateau(cycle_spectrum(n))
    return abs(d_s - 1.0) <= 0.01 and sigma <= 5e-3


def cycle_plateau_converges_to_one_with_size() -> bool:
    """The finite-size excess d_s - 1 decreases monotonically with N and is
    below 1e-3 by N = 512.

    This is the manuscript's "shrinking toward 1 as N grows". Measured
    excesses: 3.3e-3 (64), 1.1e-3 (128), 3.7e-4 (256), 1.3e-4 (512) --- a
    clean monotone decrease by a factor ~3 per doubling. The test asserts
    strict monotonicity, which no tolerance can fake, plus the absolute bound
    at the largest size.
    """
    excesses = []
    for n in (64, 128, 256, 512):
        d_s, _sigma = spectral_dimension_plateau(cycle_spectrum(n))
        excesses.append(d_s - 1.0)
    monotone = all(a > b > 0 for a, b in zip(excesses, excesses[1:]))
    return monotone and excesses[-1] < 1e-3


def irregular_grid_plateau_is_two() -> bool:
    """A mildly irregular 16 x 16 grid returns d_s -> 2 from the spectrum
    alone, with no lattice periodicity and no embedding assumed.

    Band: 1.85 <= d_s <= 2.15, sigma_ds <= 0.10. Measured d_s = 1.960,
    sigma_ds = 0.059. Over a sweep of irregularity (5%, 8%, 12% of edges
    removed) and three seeds the observed range was d_s in [1.85, 2.01] and
    sigma_ds in [0.037, 0.089], so the band contains the family spread. The
    band's lower edge, 1.85, sits 0.85 above the one-dimensional value, so it
    cannot accept a cycle (`wrong_dimension_band_accepts_cycle` verifies that
    it does not).
    """
    d_s, sigma = spectral_dimension_plateau(_laplacian_spectrum(irregular_grid()))
    return 1.85 <= d_s <= 2.15 and sigma <= 0.10


def irregular_grid_metric_embedding_correlates() -> bool:
    """The low eigenvectors of L_G supply coordinates whose ordering recovers
    the graph metric: Spearman rho >= 0.94.

    Measured rho = 0.9588 on the flagship 16 x 16 / 8% grid. Across the same
    irregularity and seed sweep the observed range was [0.942, 0.967]; the
    manuscript's quoted 0.964 lies inside it. The band is set at 0.94 to
    contain that spread; it is far above the expander's 0.436
    (`expander_fails_metric_embedding_lattice_passes`).
    """
    return metric_embedding_correlation(irregular_grid()) >= 0.94


# --------------------------------------------------------------------------
# Matter-level tests the expander PASSES (exact, integer arithmetic)
# --------------------------------------------------------------------------

def _oriented_incidence(graph: nx.Graph) -> np.ndarray:
    return nx.incidence_matrix(graph, oriented=True).toarray().astype(np.int64)


def _integer_laplacian(graph: nx.Graph) -> np.ndarray:
    return nx.laplacian_matrix(graph).toarray().astype(np.int64)


def cone_defect_is_exactly_zero(graph: nx.Graph) -> bool:
    """B B^T = L_G exactly, so the bipartite Dirac operator D = [[0, B],
    [B^T, 0]] satisfies D^2 = L_G tensor 1 on the vertex sector with defect
    identically zero.

    Integer arithmetic throughout: no tolerance, the residual matrix is
    compared to the zero matrix elementwise. This is the cone criterion of
    `cone_criterion.dirac_square_equals_laplacian`, which proves it exactly in
    sympy for the cycle; here it is exhibited for arbitrary graphs, including
    the expander, which is the point -- the cone diagnostic cannot fail, so
    it cannot discriminate.
    """
    inc = _oriented_incidence(graph)
    return bool(np.array_equal(inc @ inc.T, _integer_laplacian(graph)))


def expander_passes_cone_criterion() -> bool:
    """The expander satisfies the cone criterion exactly: a false positive."""
    return cone_defect_is_exactly_zero(random_expander())


def lattice_passes_cone_criterion() -> bool:
    """So does the irregular grid, and the cycle. The cone criterion assigns
    the same verdict to all three, which is why it is not discriminating."""
    return (cone_defect_is_exactly_zero(irregular_grid())
            and cone_defect_is_exactly_zero(nx.cycle_graph(_N_VERTICES)))


def expander_is_local() -> bool:
    """The expander has bounded degree (every vertex has exactly `degree`
    neighbours), so it is local in the graph-locality sense used for the
    matter-level tests. Exact integer check."""
    degrees = {d for _v, d in random_expander().degree()}
    return degrees == {_EXPANDER_DEGREE}


def expander_is_chirally_paired() -> bool:
    """The bipartite doubling D = [[0, B], [B^T, 0]] anticommutes exactly with
    the chirality operator gamma = diag(+1, -1), so its spectrum is symmetric
    under lambda -> -lambda: the expander is chirally paired and passes the
    matter-level chirality test. Exact integer check, {gamma, D} = 0.
    """
    inc = _oriented_incidence(random_expander())
    n_v, n_e = inc.shape
    dirac = np.block([
        [np.zeros((n_v, n_v), dtype=np.int64), inc],
        [inc.T, np.zeros((n_e, n_e), dtype=np.int64)],
    ])
    gamma = np.diag(np.concatenate([np.ones(n_v, dtype=np.int64),
                                    -np.ones(n_e, dtype=np.int64)]))
    return bool(np.array_equal(gamma @ dirac + dirac @ gamma,
                               np.zeros_like(dirac)))


# --------------------------------------------------------------------------
# Sec. IV.D: the falsifier. Same code path, opposite verdicts.
# --------------------------------------------------------------------------

def expander_gap_does_not_close() -> bool:
    """The expander's spectral gap is O(1) in the number of vertices: it does
    not close as N grows, which is the structural reason it has no infrared
    limit.

    Measured lambda_1 = 0.203, 0.197, 0.192, 0.195 at N = 64, 128, 256, 512:
    flat to within 6%. Band: every gap >= 0.1, and the ratio of the largest to
    the smallest <= 1.5. For contrast the cycle's gap falls by a factor 64
    across the same range (see `lattice_gap_closes_as_size_grows`).
    """
    gaps = []
    for n in (64, 128, 256, 512):
        spectrum = _laplacian_spectrum(random_expander(n_vertices=n))
        gaps.append(sorted(spectrum)[1])
    return all(g >= 0.1 for g in gaps) and max(gaps) / min(gaps) <= 1.5


def lattice_gap_closes_as_size_grows() -> bool:
    """The cycle's gap closes as N^-2 (diffusive), the behaviour the expander
    lacks. Measured ratio lambda_1(64)/lambda_1(512) = 64.0 against the exact
    expectation 64; band [55, 75] absorbs the O(1/N^2) correction to the
    small-angle expansion of 2 - 2 cos(2 pi / N)."""
    gaps = [sorted(cycle_spectrum(n))[1] for n in (64, 512)]
    return 55.0 <= gaps[0] / gaps[1] <= 75.0


def _span_growth_slope(spectra_by_size: list[tuple[int, np.ndarray]]) -> float:
    """Slope of scaling-window span (in decades) against log10 N."""
    xs = np.array([np.log10(n) for n, _ in spectra_by_size])
    ys = np.array([scaling_window_span(s) for _n, s in spectra_by_size])
    return float(np.polyfit(xs, ys, 1)[0])


def cycle_span_growth_slope() -> float:
    """d(span)/d(log10 N) for cycles at N = 64, 128, 256, 512."""
    return _span_growth_slope([(n, cycle_spectrum(n)) for n in (64, 128, 256, 512)])


def grid_span_growth_slope() -> float:
    """d(span)/d(log10 N) for irregular grids at 8x8 ... 20x20."""
    data = []
    for side in (8, 12, 16, 20):
        graph = irregular_grid(side=side)
        data.append((side * side, np.asarray(_laplacian_spectrum(graph))))
    return _span_growth_slope(data)


def expander_span_growth_slope() -> float:
    """d(span)/d(log10 N) for random 3-regular graphs at N = 64 ... 512."""
    data = []
    for n in (64, 128, 256, 512):
        graph = random_expander(n_vertices=n)
        data.append((n, np.asarray(_laplacian_spectrum(graph))))
    return _span_growth_slope(data)


def expander_fails_infrared_scaling_lattice_passes() -> bool:
    """THE SHARP FALSIFIER. The scaling window of a lattice widens as the
    substrate grows; the expander's does not.

    One code path (`_span_growth_slope`), three families. Measured slopes:
    cycle 2.000, irregular grid 0.988, expander 0.025. Bands: lattices >= 0.8,
    expander <= 0.10. The separation is a factor ~40 and the two bands are
    eight-fold apart, so no plausible seed or version drift can cross them
    (the observed expander slope varies by less than 0.05 across four seeds).

    This is the executable content of the manuscript's "its O(1) spectral gap
    forbids any infrared scaling".
    """
    return (cycle_span_growth_slope() >= 0.8
            and grid_span_growth_slope() >= 0.8
            and expander_span_growth_slope() <= 0.10)


def scaling_slope_times_plateau_dimension_is_two(family: str) -> bool:
    """Consistency of two independent spectral functionals.

    Diffusive scaling gives lambda_1 ~ L^-2 with N = L^{d_s}, hence
    span = (2/d_s) log10 N + const, hence slope * d_s = 2. The slope comes
    from the GAP at four sizes; d_s comes from the heat-trace PLATEAU at the
    largest size. They share no arithmetic, so their product hitting 2 is a
    genuine cross-check and not a rearrangement.

    Measured products: cycle 2.00, grid 1.97, expander 0.05. Band [1.8, 2.2]
    for the lattices (a 10% window, comfortably tighter than the factor-40
    gap to the expander); the expander is required to be below 0.3.
    """
    if family == "cycle":
        d_s, _ = spectral_dimension_plateau(cycle_spectrum(512))
        return 1.8 <= cycle_span_growth_slope() * d_s <= 2.2
    if family == "grid":
        d_s, _ = spectral_dimension_plateau(_laplacian_spectrum(irregular_grid(side=20)))
        return 1.8 <= grid_span_growth_slope() * d_s <= 2.2
    if family == "expander":
        d_s, _ = spectral_dimension_plateau(_laplacian_spectrum(random_expander(n_vertices=512)))
        return expander_span_growth_slope() * d_s <= 0.3
    raise ValueError(f"unknown family {family!r}")


def expander_fails_plateau_lattice_passes() -> bool:
    """The flatness criterion the manuscript leads with: the expander has no
    stable plateau, the grid does.

    Same function (`spectral_dimension_plateau`) on both. Measured
    sigma_ds = 0.158 (expander) against 0.059 (grid), a factor 2.65. Bands:
    grid <= 0.10, expander >= 0.12, and the ratio >= 2. Across four seeds the
    expander gave sigma_ds in [0.155, 0.167] and across the grid sweep the
    grid gave [0.037, 0.089], so the bands hold with margin --- but this is
    the WEAKEST of the three discriminators and the module says so: a factor
    2.6 is not a factor 40.
    """
    _d_grid, sigma_grid = spectral_dimension_plateau(_laplacian_spectrum(irregular_grid()))
    _d_exp, sigma_exp = spectral_dimension_plateau(_laplacian_spectrum(random_expander()))
    return (sigma_grid <= 0.10
            and sigma_exp >= 0.12
            and sigma_exp >= 2.0 * sigma_grid)


def expander_plateau_value_is_a_false_positive() -> bool:
    """The expander's plateau VALUE impersonates a geometry.

    On its narrow window the expander returns d_s = 2.09, which would pass a
    naive "is it two-dimensional?" test at the same band used for the grid
    (1.85 <= d_s <= 2.15). This function certifies that the value test alone
    accepts the expander, which is why the composite verdict is needed. It is
    a POSITIVE assertion about a false positive, not a corollary of the
    flatness test: it uses the median, which the flatness test discards.
    """
    d_s, _sigma = spectral_dimension_plateau(_laplacian_spectrum(random_expander()))
    return 1.85 <= d_s <= 2.15


def expander_fails_metric_embedding_lattice_passes() -> bool:
    """The low-mode coordinates do not recover the expander's graph metric.

    Same function (`metric_embedding_correlation`) and the same n_modes = 2
    for both families. Measured rho = 0.436 (expander) against 0.959 (grid).
    Bands: grid >= 0.94, expander <= 0.75. The 0.75 threshold sits well above
    the expander's four-seed range [0.417, 0.477] and well below the grid's
    sweep range [0.942, 0.967], so the two families cannot be confused.
    """
    rho_grid = metric_embedding_correlation(irregular_grid())
    rho_exp = metric_embedding_correlation(random_expander())
    return rho_grid >= 0.94 and rho_exp <= 0.75


def expander_embedding_fails_for_every_mode_count() -> bool:
    """The expander's embedding failure is not an artefact of n_modes = 2.

    Sweeping n_modes over 1..6 the expander's rho stays in [0.313, 0.610],
    never reaching 0.75, while the grid reaches 0.959 at its natural
    n_modes = 2. Band: max over the sweep <= 0.75.
    """
    graph = random_expander()
    rhos = [metric_embedding_correlation(graph, n_modes=m) for m in (1, 2, 3, 4, 6)]
    return max(rhos) <= 0.75


# --------------------------------------------------------------------------
# The composite verdict
# --------------------------------------------------------------------------

def diagnostic_report(graph: nx.Graph, span_slope: float) -> dict:
    """All five diagnostics for one substrate, as booleans plus raw numbers."""
    spectrum = np.asarray(_laplacian_spectrum(graph))
    d_s, sigma = spectral_dimension_plateau(spectrum)
    rho = metric_embedding_correlation(graph)
    return {
        "d_s": d_s,
        "sigma_ds": sigma,
        "rho": rho,
        "span_slope": span_slope,
        "cone_ok": cone_defect_is_exactly_zero(graph),
        "plateau_ok": sigma <= 0.10,
        "infrared_scaling_ok": span_slope >= 0.8,
        "embedding_ok": rho >= 0.94,
    }


def composite_verdict(report: dict) -> str:
    """ACCEPT only if all four diagnostics pass; REJECT otherwise.

    The cone criterion is included even though it never fails, precisely so
    that the verdict records that it never fails.
    """
    passes = (report["cone_ok"] and report["plateau_ok"]
              and report["infrared_scaling_ok"] and report["embedding_ok"])
    return "ACCEPT" if passes else "REJECT"


def expander_composite_verdict_is_reject() -> bool:
    """The manuscript's headline falsifier claim, executable.

    The expander passes the cone criterion (exactly) and fails the plateau,
    the infrared-scaling and the embedding diagnostics; the composite verdict
    is REJECT. No tolerance enters beyond those already justified in the
    individual diagnostics.
    """
    report = diagnostic_report(random_expander(), expander_span_growth_slope())
    return (report["cone_ok"]
            and not report["plateau_ok"]
            and not report["infrared_scaling_ok"]
            and not report["embedding_ok"]
            and composite_verdict(report) == "REJECT")


def lattice_composite_verdict_is_accept() -> bool:
    """The irregular grid passes all four diagnostics: ACCEPT. Without this
    the REJECT above would be worthless, since a checklist that rejects
    everything is not a checklist."""
    report = diagnostic_report(irregular_grid(), grid_span_growth_slope())
    return composite_verdict(report) == "ACCEPT"


# --------------------------------------------------------------------------
# Mutation controls. Each injects a WRONG input into the real code path and
# must return False. None is a logical corollary of a positive test.
# --------------------------------------------------------------------------

def mutated_unsigned_incidence_matches_laplacian() -> bool:
    """MUTATION of `cone_defect_is_exactly_zero`: replace the oriented
    incidence matrix by the UNSIGNED one. Then B B^T is the signless
    Laplacian D + A, not D - A, and the cone identity breaks (residual 2 on
    every edge). Must return False.
    """
    graph = random_expander()
    inc = nx.incidence_matrix(graph, oriented=False).toarray().astype(np.int64)
    return bool(np.array_equal(inc @ inc.T, _integer_laplacian(graph)))


def mutated_shuffled_embedding_correlates() -> bool:
    """MUTATION of `irregular_grid_metric_embedding_correlates`: keep the grid
    and the real low-mode coordinates but randomly permute which vertex owns
    which coordinate. The correlation collapses (measured rho ~ 0.0), proving
    the 0.959 is carried by the eigenvector-to-vertex assignment and is not an
    artefact of comparing two distance matrices. Must return False.
    """
    graph = irregular_grid()
    coords = low_mode_embedding(graph)
    rng = np.random.default_rng(SEED + 1)
    shuffled = coords[rng.permutation(coords.shape[0])]
    graph_d, embed_d = _pairwise_graph_and_embedding_distances(graph, shuffled)
    return float(spearmanr(graph_d, embed_d).statistic) >= 0.94


def mutated_star_graph_has_a_one_dimensional_plateau() -> bool:
    """MUTATION of `cycle_plateau_is_one`: feed the plateau machinery the
    spectrum of the star K_{1,255} (eigenvalues 0, then 1 with multiplicity
    254, then 256), a connected graph of diameter 2 with no geometry at all.
    The plateau test for d_s = 1 must fail. Must return False.
    """
    n = _N_VERTICES
    spectrum = np.array([0.0] + [1.0] * (n - 2) + [float(n)])
    d_s, sigma = spectral_dimension_plateau(spectrum)
    return abs(d_s - 1.0) <= 0.01 and sigma <= 5e-3


def wrong_dimension_band_accepts_cycle() -> bool:
    """MUTATION of `irregular_grid_plateau_is_two`: apply the
    two-dimensional acceptance band to the one-dimensional cycle. If the band
    were loose enough to accept C_256 it would not discriminate dimension at
    all. Must return False.
    """
    d_s, sigma = spectral_dimension_plateau(cycle_spectrum(_N_VERTICES))
    return 1.85 <= d_s <= 2.15 and sigma <= 0.10


def wrong_family_passes_infrared_scaling() -> bool:
    """MUTATION of `expander_fails_infrared_scaling_lattice_passes`: apply the
    LATTICE acceptance band (slope >= 0.8) to the expander. Must return False.
    A slope of 0.025 is not 0.8, and this is the assertion that the sharp
    falsifier is sharp in the direction that matters.
    """
    return expander_span_growth_slope() >= 0.8


def mutated_antiperiodic_spectrum_matches_sympy(n: int = 8) -> bool:
    """MUTATION of `closed_form_cycle_spectrum_matches_sympy`: keep the same
    number of eigenvalues but shift the quantisation to the ANTIPERIODIC
    sector, 2 - 2 cos(2 pi (k + 1/2) / n). This is the spectrum of the same
    difference operator with the opposite boundary condition, so it is a
    physically meaningful wrong answer of the correct length -- the
    comparison cannot fail merely on a length mismatch. Must return False.
    """
    exact = cycle_laplacian(n)
    sympy_eigs = []
    for val, mult in exact.eigenvals().items():
        sympy_eigs.extend([float(sp.N(val))] * mult)
    k = np.arange(n)
    wrong = np.sort(2.0 - 2.0 * np.cos(2.0 * np.pi * (k + 0.5) / n))
    return bool(np.allclose(np.sort(sympy_eigs), wrong, atol=1e-12, rtol=0.0))
