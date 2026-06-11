from validators.tetrad_from_step import (
    graded_step_has_nonzero_tetrad_gradient,
    homogeneous_step_is_flat,
    inverse_tetrad_reads_off_velocities,
    squared_hamiltonian_gives_pointwise_metric,
)


def test_squared_hamiltonian_metric():
    assert squared_hamiltonian_gives_pointwise_metric()


def test_inverse_tetrad():
    assert inverse_tetrad_reads_off_velocities()


def test_homogeneous_flat():
    assert homogeneous_step_is_flat()


def test_graded_nonzero_gradient():
    assert graded_step_has_nonzero_tetrad_gradient()
