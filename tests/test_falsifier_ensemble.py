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
