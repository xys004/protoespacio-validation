from validators.tetrad_from_step import (
    centered_link_error_is_second_order,
    conformal_tetrad_matches_spin_connection_input,
    discrete_link_continuum_limit_is_symmetrized,
    discrete_link_hamiltonian_is_hermitian,
    graded_step_has_nonzero_tetrad_gradient,
    homogeneous_step_is_flat,
    inverse_tetrad_reads_off_velocities,
    squared_hamiltonian_gives_pointwise_metric,
)


def test_discrete_link_hamiltonian_is_hermitian():
    assert discrete_link_hamiltonian_is_hermitian()


def test_discrete_link_limit_is_symmetrized_continuum_operator():
    assert discrete_link_continuum_limit_is_symmetrized()


def test_centered_link_error_starts_at_second_order():
    assert centered_link_error_is_second_order()


def test_frozen_symbol_gives_pointwise_metric():
    assert squared_hamiltonian_gives_pointwise_metric()


def test_inverse_tetrad():
    assert inverse_tetrad_reads_off_velocities()


def test_homogeneous_flat():
    assert homogeneous_step_is_flat()


def test_graded_nonzero_gradient():
    assert graded_step_has_nonzero_tetrad_gradient()


def test_conformal_tetrad_matches_spin_connection_input():
    assert conformal_tetrad_matches_spin_connection_input()
