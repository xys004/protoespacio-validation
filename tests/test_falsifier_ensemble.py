from validators.falsifier_ensemble import (
    discriminating_diagnostics_do_not_overlap,
    ensemble_is_the_advertised_size,
    every_expander_is_rejected_and_every_lattice_accepted,
    ladder_sides_are_even_as_the_expander_requires,
    margins_match_the_quoted_values,
    mutated_recomputation_against_wrong_seed_matches,
    mutated_swapped_families_still_separate,
    spectral_dimension_alone_does_not_separate,
    stored_ensemble_reproduces_under_current_code,
    what_separates_is_plateau_stability_not_its_value,
    expander_rejection_is_perfect_at_every_size_and_degree,
    lattice_acceptance_is_perfect_only_above_the_smallest_size,
    mutated_scale_margins_read_backwards_still_grow,
    separation_margins_grow_with_system_size,
    spectral_dimension_ranges_overlap_across_sizes,
    the_small_size_failure_is_a_threshold_margin_effect,
)


def test_ensemble_size():
    """800 instances, 400 per family, none errored, all at N = 1936."""
    assert ensemble_is_the_advertised_size()


def test_perfect_verdict_separation():
    """400/400 expanders REJECT and 400/400 lattices ACCEPT."""
    assert every_expander_is_rejected_and_every_lattice_accepted()


def test_diagnostics_do_not_overlap():
    """Worst lattice beats best expander in rho, span slope and sigma_ds separately."""
    assert discriminating_diagnostics_do_not_overlap()


def test_margins_match_quoted():
    """The margins 0.71 (rho), 0.74 (slope) and 0.10 (sigma_ds) quoted in the text."""
    assert margins_match_the_quoted_values()


def test_spectral_dimension_alone_fails():
    """d_s alone is not a usable discriminator: its gap is a fraction of the others."""
    assert spectral_dimension_alone_does_not_separate()


def test_stability_not_value_is_what_separates():
    """The expander's plateau value is lattice-like; its stability is 2.6x worse."""
    assert what_separates_is_plateau_stability_not_its_value()


def test_stored_ensemble_still_matches_the_code():
    """INTEGRITY: instances rebuilt from their seeds reproduce the stored record."""
    assert stored_ensemble_reproduces_under_current_code()


def test_ladder_sides_even():
    """Every ladder size admits a cubic random regular graph (n*d even)."""
    assert ladder_sides_are_even_as_the_expander_requires()


def test_mutation_swapped_families():
    """MUTATION: exchanging the family labels breaks the separation claim."""
    assert not mutated_swapped_families_still_separate()


def test_mutation_wrong_seed():
    """MUTATION: the integrity check fails when matched against a different seed."""
    assert not mutated_recomputation_against_wrong_seed_matches()


# --- the scale-up: three sizes x three expander degrees ---------------------

def test_expander_rejection_perfect_at_every_size_and_degree():
    """1350/1350 expanders REJECT across N = 256, 1024, 2304 and degrees 3, 4, 6."""
    assert expander_rejection_is_perfect_at_every_size_and_degree()


def test_lattice_acceptance_needs_size():
    """Lattices: 150/150 at N >= 1024 but 149/150 at N = 256 -- the criterion needs size."""
    assert lattice_acceptance_is_perfect_only_above_the_smallest_size()


def test_small_size_failure_is_a_threshold_margin():
    """The one rejected lattice misses the embedding threshold by 0.0011, nothing else."""
    assert the_small_size_failure_is_a_threshold_margin_effect()


def test_margins_grow_with_size():
    """Metric-correlation margin 0.394 -> 0.614 -> 0.734: the separation widens with N."""
    assert separation_margins_grow_with_system_size()


def test_spectral_dimension_ranges_overlap():
    """Pooled over sizes the d_s ranges overlap outright: not a discriminator at all."""
    assert spectral_dimension_ranges_overlap_across_sizes()


def test_mutation_scale_margins_backwards():
    """MUTATION: margins with the families exchanged are negative, so growth fails."""
    assert not mutated_scale_margins_read_backwards_still_grow()
