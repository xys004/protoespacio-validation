"""Tests for the geometry diagnostics of Sec. IV.C-IV.D.

Positive tests certify the calibrated diagnostics (cycle plateau at d_s = 1,
irregular grid at d_s = 2, low-mode metric embedding) and the expander
falsifier. Mutation controls inject a wrong input into the same code path and
must fail; none of them is a logical corollary of the corresponding positive
test.
"""
import pytest

from validators.geometry_diagnostics import (
    closed_form_cycle_spectrum_matches_sympy,
    cycle_plateau_converges_to_one_with_size,
    cycle_plateau_is_one,
    expander_composite_verdict_is_reject,
    expander_embedding_fails_for_every_mode_count,
    expander_fails_infrared_scaling_lattice_passes,
    expander_fails_metric_embedding_lattice_passes,
    expander_fails_plateau_lattice_passes,
    expander_gap_does_not_close,
    expander_is_chirally_paired,
    expander_is_local,
    expander_passes_cone_criterion,
    expander_plateau_value_is_a_false_positive,
    irregular_grid_metric_embedding_correlates,
    irregular_grid_plateau_is_two,
    lattice_composite_verdict_is_accept,
    lattice_gap_closes_as_size_grows,
    lattice_passes_cone_criterion,
    mutated_antiperiodic_spectrum_matches_sympy,
    mutated_shuffled_embedding_correlates,
    mutated_star_graph_has_a_one_dimensional_plateau,
    mutated_unsigned_incidence_matches_laplacian,
    scaling_slope_times_plateau_dimension_is_two,
    wrong_dimension_band_accepts_cycle,
    wrong_family_passes_infrared_scaling,
)


# --------------------------------------------------------------------------
# Bridge to the exact sympy layer
# --------------------------------------------------------------------------

def test_closed_form_cycle_spectrum_matches_sympy():
    """The numeric cycle spectrum used throughout agrees with the exact sympy
    Laplacian of `laplacian_projector.cycle_laplacian` to 1e-12."""
    assert closed_form_cycle_spectrum_matches_sympy()


# --------------------------------------------------------------------------
# Sec. IV.C: the calibrated diagnostic
# --------------------------------------------------------------------------

def test_cycle_plateau_is_one():
    """Calibration: the heat-kernel spectral dimension of C_256 plateaus at
    d_s = 1.00037 with sigma_ds = 4.5e-4, the exactly known answer."""
    assert cycle_plateau_is_one()


def test_cycle_plateau_converges_to_one_with_size():
    """The finite-size excess d_s - 1 decreases monotonically over
    N = 64, 128, 256, 512 and is below 1e-3 at N = 512."""
    assert cycle_plateau_converges_to_one_with_size()


def test_irregular_grid_plateau_is_two():
    """A mildly irregular 16x16 grid returns d_s = 1.96, reconstructed from
    the spectrum of L_G alone with no embedding assumed."""
    assert irregular_grid_plateau_is_two()


def test_irregular_grid_metric_embedding_correlates():
    """The first two non-trivial eigenvectors of L_G give coordinates whose
    ordering recovers the graph metric at Spearman rho = 0.959."""
    assert irregular_grid_metric_embedding_correlates()


# --------------------------------------------------------------------------
# Matter-level tests the expander passes (exact integer arithmetic)
# --------------------------------------------------------------------------

def test_expander_passes_cone_criterion():
    """The expander satisfies the cone criterion with defect identically
    zero: an exact false positive, not merely a small one."""
    assert expander_passes_cone_criterion()


def test_lattice_passes_cone_criterion():
    """So do the irregular grid and the cycle, which is why the cone
    diagnostic cannot discriminate between them."""
    assert lattice_passes_cone_criterion()


def test_expander_is_local():
    """The expander has bounded degree, hence is graph-local."""
    assert expander_is_local()


def test_expander_is_chirally_paired():
    """The expander's bipartite Dirac doubling anticommutes exactly with the
    chirality operator, so its spectrum is chirally paired."""
    assert expander_is_chirally_paired()


# --------------------------------------------------------------------------
# Sec. IV.D: the falsifier
# --------------------------------------------------------------------------

def test_expander_gap_does_not_close():
    """The expander's spectral gap stays O(1) over N = 64...512 (flat to 6%),
    the structural reason it has no infrared limit."""
    assert expander_gap_does_not_close()


def test_lattice_gap_closes_as_size_grows():
    """The cycle's gap falls by the diffusive factor 64 over the same range,
    the behaviour the expander lacks."""
    assert lattice_gap_closes_as_size_grows()


def test_expander_fails_infrared_scaling_lattice_passes():
    """Sharp falsifier: the scaling window widens with N for the cycle (slope
    2.00) and the grid (0.99) but not for the expander (0.025), all three
    measured by one code path."""
    assert expander_fails_infrared_scaling_lattice_passes()


@pytest.mark.parametrize("family", ["cycle", "grid", "expander"])
def test_scaling_slope_times_plateau_dimension_is_two(family):
    """Cross-check between two independent spectral functionals: the
    gap-scaling slope times the heat-trace plateau dimension equals 2 for
    both lattices (2.00, 1.97) and collapses to 0.05 for the expander."""
    assert scaling_slope_times_plateau_dimension_is_two(family)


def test_expander_fails_plateau_lattice_passes():
    """The expander has no stable plateau: sigma_ds = 0.158 against the
    grid's 0.059. This is the weakest of the three discriminators."""
    assert expander_fails_plateau_lattice_passes()


def test_expander_plateau_value_is_a_false_positive():
    """The expander's plateau VALUE d_s = 2.09 passes the same
    two-dimensionality band the grid passes: the value, like the cone defect,
    is a false positive."""
    assert expander_plateau_value_is_a_false_positive()


def test_expander_fails_metric_embedding_lattice_passes():
    """Same embedding code and the same n_modes = 2 give rho = 0.436 for the
    expander against 0.959 for the grid."""
    assert expander_fails_metric_embedding_lattice_passes()


def test_expander_embedding_fails_for_every_mode_count():
    """The embedding failure is not an artefact of n_modes = 2: over
    n_modes = 1, 2, 3, 4, 6 the expander never exceeds rho = 0.61."""
    assert expander_embedding_fails_for_every_mode_count()


def test_expander_composite_verdict_is_reject():
    """The headline falsifier: the expander passes the cone criterion and
    fails plateau, infrared scaling and embedding, so the composite verdict
    is REJECT."""
    assert expander_composite_verdict_is_reject()


def test_lattice_composite_verdict_is_accept():
    """The irregular grid passes all four diagnostics, so the checklist is
    not one that rejects everything."""
    assert lattice_composite_verdict_is_accept()


# --------------------------------------------------------------------------
# Mutation controls: each must be False
# --------------------------------------------------------------------------

def test_mutated_unsigned_incidence_does_not_match_laplacian():
    """Replacing the oriented incidence matrix by the unsigned one gives the
    signless Laplacian, and the cone identity breaks."""
    assert not mutated_unsigned_incidence_matches_laplacian()


def test_mutated_shuffled_embedding_loses_correlation():
    """Permuting which vertex owns which low-mode coordinate collapses the
    correlation to rho = 0.011, so the grid's 0.959 is carried by the
    eigenvector-to-vertex assignment and not by comparing two distance
    matrices."""
    assert not mutated_shuffled_embedding_correlates()


def test_mutated_star_graph_has_no_one_dimensional_plateau():
    """The star K_{1,255} -- connected, diameter 2, no geometry -- is
    rejected by the same plateau test that accepts the cycle."""
    assert not mutated_star_graph_has_a_one_dimensional_plateau()


def test_wrong_dimension_band_rejects_cycle():
    """The two-dimensional acceptance band rejects the one-dimensional cycle,
    so the band genuinely discriminates dimension."""
    assert not wrong_dimension_band_accepts_cycle()


def test_wrong_family_fails_infrared_scaling():
    """The lattice acceptance band on the infrared-scaling slope rejects the
    expander, which is the direction in which the falsifier must be sharp."""
    assert not wrong_family_passes_infrared_scaling()


def test_mutated_antiperiodic_spectrum_does_not_match_sympy():
    """The antiperiodic quantisation gives the same number of eigenvalues but
    the wrong ones, so the sympy cross-check is not length-blind."""
    assert not mutated_antiperiodic_spectrum_matches_sympy()
