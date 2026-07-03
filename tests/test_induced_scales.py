from validators.induced_scales import (
    a0_term_produces_Lambda4,
    a1_term_produces_Lambda2,
    a2_term_produces_only_log_Lambda,
    a3_term_is_cutoff_finite,
    corner_reduction_is_justified,
    hard_cutoff_scheme_constants_are_derived,
    implied_cutoff_is_planckian,
    pi_bracket_is_certified,
    scheme_constant_varies_but_power_is_invariant,
    mutated_cutoff_power_matches,
    mutated_dimension_prefactor_matches,
    mutated_measure_matches,
    mutated_vacuum_scale_matches,
    vacuum_overshoot_is_ten_to_120_class,
)


def test_a0_lambda4():
    """The cosmological term scales as Lambda^4: exact integral."""
    assert a0_term_produces_Lambda4()


def test_a1_lambda2():
    """The Einstein-Hilbert term scales as Lambda^2: exact integral."""
    assert a1_term_produces_Lambda2()


def test_a2_log_only():
    """The a_2 term produces only log(Lambda): exact integral."""
    assert a2_term_produces_only_log_Lambda()


def test_a3_cutoff_finite():
    """Terms beyond a_2 are cutoff-finite."""
    assert a3_term_is_cutoff_finite()


def test_hard_cutoff_scheme_constants():
    """Hard-cutoff constants 1/(64 pi^2), 1/(32 pi^2) are derived outputs."""
    assert hard_cutoff_scheme_constants_are_derived()


def test_scheme_constant_vs_power():
    """Regulator moves the O(1) constant but never the Lambda^2 power."""
    assert scheme_constant_varies_but_power_is_invariant()


def test_mutation_dimension_prefactor():
    """A d=2 prefactor slip gives log instead of Lambda^2: check must fail."""
    assert not mutated_dimension_prefactor_matches()


def test_mutation_measure():
    """Dropping the 1/s of the proper-time measure breaks the Lambda^4 check."""
    assert not mutated_measure_matches()


def test_pi_bracket():
    """31415/10000 < pi < 31416/10000, the only non-rational input."""
    assert pi_bracket_is_certified()


def test_corner_reduction():
    """Closed form of the overshoot ratio + monotonicity justify corner arithmetic."""
    assert corner_reduction_is_justified()


def test_cutoff_is_planckian():
    """M_Pl < Lambda < 20 M_Pl over the whole (N, c) window: Planck scale as output."""
    assert implied_cutoff_is_planckian()


def test_vacuum_overshoot_1e120():
    """10^121 < rho_vac/rho_obs < 10^127 over the window: the CC problem, executable."""
    assert vacuum_overshoot_is_ten_to_120_class()


def test_mutation_cutoff_power():
    """The wrong matching power 1/G = N c Lambda leaves the Planck band."""
    assert not mutated_cutoff_power_matches()


def test_mutation_vacuum_scale():
    """A meV -> GeV slip in the observed vacuum scale leaves the overshoot band."""
    assert not mutated_vacuum_scale_matches()
