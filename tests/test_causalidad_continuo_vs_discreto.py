from validators.causalidad_continuo_vs_discreto import (
    continuous_dispersion_expansion_matches_trace_limit,
    lieb_robinson_bound_holds,
    small_step_dispersion_recovers_continuous,
)


def test_continuous_expansion():
    assert continuous_dispersion_expansion_matches_trace_limit()


def test_small_step_recovery():
    assert small_step_dispersion_recovers_continuous()


def test_lieb_robinson():
    assert lieb_robinson_bound_holds()
