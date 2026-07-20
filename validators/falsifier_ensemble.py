"""Ensemble certification of the geometry falsifier.

`geometry_diagnostics.py` runs the composite criterion on ONE lattice and ONE
expander at one seed and N = 256. That certifies the criterion is executable; it
does not certify that it SEPARATES the two families, because a single instance
of each cannot exhibit a margin. Rejecting one expander falsifies a checklist,
not a programme.

This module closes that gap using a precomputed ensemble: 400 independent
instances of each family, each carried up a size ladder N = 256, 484, 1024,
1936, produced by `results/expander_ensemble.py` on a 32-thread workstation
(~30 min wall) and stored in `results/expander_ensemble.json`. The ensemble is
not reproduced here -- it costs far more than a test suite may -- so this module
does three things instead:

  1. CERTIFIES THE CLAIMS the manuscript makes from the ensemble: that the
     composite verdict separates the families perfectly, and that the three
     discriminating diagnostics have a positive margin with no overlap.

  2. CERTIFIES A NEGATIVE that matters: the spectral dimension d_s ALONE does
     not usefully separate them. The expander does display a plateau near 2. A
     one-diagnostic version of the criterion would pass it. This is why the
     criterion is composite, and it is the sharpest thing the ensemble taught
     us that the single-instance module could not see.

  3. GUARDS AGAINST DRIFT. Stored results are worthless if the code that
     produced them has since changed; a JSON file is not a certificate. So
     `stored_ensemble_reproduces_under_current_code` rebuilds sampled instances
     from their seeds with TODAY'S `geometry_diagnostics` and requires the
     recomputed diagnostics to match the stored record. If a future edit changes
     what the diagnostics mean, this fails and the manuscript's ensemble numbers
     are known to be stale.

The ladder sides are even (16, 22, 32, 44) because a cubic random regular graph
requires n*d even -- a constraint invisible at the manuscript's single N = 256
and discovered only on scaling up.

Sustains:
- master_protospace.tex, Sec. IV.D (the expander falsifier)
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np

from validators.geometry_diagnostics import (
    _laplacian_spectrum,
    irregular_grid,
    random_expander,
    scaling_window_span,
)

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENSEMBLE = os.path.join(_HERE, "results", "expander_ensemble.json")

# Must match results/expander_ensemble.py exactly; asserted below.
_SIDES = (16, 22, 32, 44)
_SIZES = tuple(s * s for s in _SIDES)
_TOP_N = _SIZES[-1]
_GRID_REMOVED = 0.08
_EXPANDER_DEGREE = 3


@lru_cache(maxsize=1)
def _records() -> tuple:
    with open(_ENSEMBLE, encoding="utf-8") as fh:
        return tuple(json.load(fh)["records"])


def _by(family: str, key: str) -> list:
    return [r[key] for r in _records() if r["family"] == family]


# ---------------------------------------------------------------------------
# 1. The claims the manuscript makes from the ensemble
# ---------------------------------------------------------------------------

def ensemble_is_the_advertised_size() -> bool:
    """400 instances per family, 800 total, no errored instance, all carried to
    N = 1936. If the stored file is truncated or padded the manuscript's counts
    are wrong, so this is checked before anything is read off it."""
    recs = _records()
    fams = {"expander": 0, "grid": 0}
    for r in recs:
        if "error" in r:
            return False
        if r["family"] not in fams or r["n_vertices"] != _TOP_N:
            return False
        fams[r["family"]] += 1
    return len(recs) == 800 and fams["expander"] == 400 and fams["grid"] == 400


def every_expander_is_rejected_and_every_lattice_accepted() -> bool:
    """The headline: 400/400 REJECT and 400/400 ACCEPT. A checklist that
    rejected everything, or accepted everything, would be worthless; both halves
    are required."""
    exp = [r["verdict"] for r in _records() if r["family"] == "expander"]
    lat = [r["verdict"] for r in _records() if r["family"] == "grid"]
    return (len(exp) == 400 and all(v == "REJECT" for v in exp)
            and len(lat) == 400 and all(v == "ACCEPT" for v in lat))


def discriminating_diagnostics_do_not_overlap() -> bool:
    """The three discriminating diagnostics have a positive margin: the WORST
    lattice is still better than the BEST expander, for each of them
    separately. Margins measured: rho 0.707, span slope 0.738, sigma_ds 0.099.

    This is the quantitative content of "separated": not a difference of means,
    which any two families have, but disjoint ranges over 800 instances."""
    return (min(_by("grid", "rho")) > max(_by("expander", "rho"))
            and min(_by("grid", "span_slope")) > max(_by("expander", "span_slope"))
            and max(_by("grid", "sigma_ds")) < min(_by("expander", "sigma_ds")))


def margins_match_the_quoted_values() -> bool:
    """The three margins quoted in the manuscript, to two decimals."""
    m_rho = min(_by("grid", "rho")) - max(_by("expander", "rho"))
    m_slope = min(_by("grid", "span_slope")) - max(_by("expander", "span_slope"))
    m_sigma = min(_by("expander", "sigma_ds")) - max(_by("grid", "sigma_ds"))
    return (abs(m_rho - 0.71) < 0.005
            and abs(m_slope - 0.74) < 0.005
            and abs(m_sigma - 0.10) < 0.005)


# ---------------------------------------------------------------------------
# 2. The negative result: d_s alone is not a criterion
# ---------------------------------------------------------------------------

def spectral_dimension_alone_does_not_separate() -> bool:
    """The expander's spectral-dimension plateau sits near 2, like the lattice's:
    means 2.081 vs 2.056. The d_s ranges are ordered but the gap is ~0.02,
    twenty times smaller than the sigma_ds margin and thirty times smaller than
    the rho margin -- far inside the spread a different graph family or size
    would produce, and therefore not a usable single discriminator.

    Certified as an INEQUALITY between margins rather than as an overlap, which
    is the honest statement: the families are ordered in d_s, but not separated
    by it in any sense a criterion could rely on."""
    lat, exp = _by("grid", "d_s"), _by("expander", "d_s")
    ds_gap = min(exp) - max(lat)
    rho_margin = min(_by("grid", "rho")) - max(_by("expander", "rho"))
    sigma_margin = min(_by("expander", "sigma_ds")) - max(_by("grid", "sigma_ds"))
    return (abs(ds_gap) < 0.1 * sigma_margin
            and abs(ds_gap) < 0.05 * rho_margin
            and 1.9 < float(np.mean(exp)) < 2.2)


def what_separates_is_plateau_stability_not_its_value() -> bool:
    """Restating the previous result constructively: the expander's plateau VALUE
    is lattice-like, while its plateau STABILITY sigma_ds is 2.6x worse. The
    diagnostic that works is the one measuring whether a plateau exists, not
    what it reads."""
    ratio = float(np.mean(_by("expander", "sigma_ds"))
                  / np.mean(_by("grid", "sigma_ds")))
    ds_ratio = float(np.mean(_by("expander", "d_s")) / np.mean(_by("grid", "d_s")))
    return ratio > 2.0 and abs(ds_ratio - 1.0) < 0.05


# ---------------------------------------------------------------------------
# 3. Anti-drift: the stored file must still describe today's code
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _recomputed_spans(family: str, seed: int, upto: int) -> tuple:
    """Rebuild an instance from its seed with TODAY'S code and return the
    scaling-window span at each ladder size up to `upto`.

    Only the two smallest rungs are used. The full ladder is not recomputed:
    at N = 1936 the embedding correlation alone needs all-pairs graph distances
    and costs minutes, which a test suite may not spend. What IS recomputed is
    the part that everything else is downstream of -- the seeded graph
    construction and the Laplacian spectrum -- so a change in either is caught.
    A change confined to the embedding or plateau estimators downstream of the
    spectrum would not be; that residual is stated rather than hidden."""
    out = []
    for side, n in zip(_SIDES, _SIZES):
        if n > upto:
            break
        g = (random_expander(_EXPANDER_DEGREE, n, seed) if family == "expander"
             else irregular_grid(side, _GRID_REMOVED, seed))
        out.append((n, float(scaling_window_span(np.asarray(_laplacian_spectrum(g))))))
    return tuple(out)


def stored_ensemble_reproduces_under_current_code() -> bool:
    """THE INTEGRITY CERTIFICATE. For one instance of each family, rebuild the
    graphs from their seeds with today's code and require the scaling-window
    spans to match those recorded in `span_by_size` to 1e-9.

    Without this the module would certify a JSON file rather than a fact about
    the code: a later change to the graph builders, the seeding, or the
    Laplacian would leave the manuscript quoting numbers nothing in the
    repository still produces, and nothing would complain. Checked on the two
    smallest rungs for cost; see `_recomputed_spans` for what that does and does
    not cover."""
    for family, seed in (("expander", 7), ("grid", 7)):
        stored = next(r for r in _records()
                      if r["family"] == family and r["seed"] == seed)
        stored_spans = {int(n): float(v) for n, v in stored["span_by_size"]}
        for n, span in _recomputed_spans(family, seed, _SIZES[1]):
            if n not in stored_spans or abs(span - stored_spans[n]) > 1e-9:
                return False
    return True


def ladder_sides_are_even_as_the_expander_requires() -> bool:
    """A cubic random regular graph needs n*d even, so every side in the ladder
    must be even. The manuscript's own N = 256 satisfies this accidentally; the
    constraint only appears on scaling up, and violating it raises rather than
    silently biasing, so it is recorded here as a property of the design."""
    return all(s % 2 == 0 for s in _SIDES) and all((n * _EXPANDER_DEGREE) % 2 == 0
                                                   for n in _SIZES)


# ---------------------------------------------------------------------------
# Mutation controls. Each injects a WRONG input into the real code path.
# ---------------------------------------------------------------------------

def mutated_swapped_families_still_separate() -> bool:
    """MUTATION: read the verdicts with the family labels exchanged. The
    separation claim must then FAIL -- otherwise the check is insensitive to
    which family is which, i.e. it is measuring nothing. Must return False."""
    exp = [r["verdict"] for r in _records() if r["family"] == "grid"]
    lat = [r["verdict"] for r in _records() if r["family"] == "expander"]
    return all(v == "REJECT" for v in exp) and all(v == "ACCEPT" for v in lat)


def mutated_recomputation_against_wrong_seed_matches() -> bool:
    """MUTATION of the integrity certificate: compare the recomputation of seed
    7 against the STORED record of seed 8. Different seeds give different
    graphs, so this must FAIL; if it passed, the certificate would be matching
    something seed-independent and would not detect drift. Must return False."""
    stored = next(r for r in _records()
                  if r["family"] == "expander" and r["seed"] == 8)
    stored_spans = {int(n): float(v) for n, v in stored["span_by_size"]}
    return all(abs(span - stored_spans[n]) <= 1e-9
               for n, span in _recomputed_spans("expander", 7, _SIZES[1]))
