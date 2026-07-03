from validators.spinor_curvature import (
    cartan_curvature_matches_christoffel_riemann,
    flat_background_commutator_vanishes,
    mutated_connection_matches,
    mutated_normalization_one_eighth_matches,
    mutated_normalization_one_half_matches,
    mutated_sign_matches,
    spin_generator_matches_lorentz_sigma,
    spinor_commutator_is_quarter_riemann,
)


def test_spin_generator_matches_lorentz_sigma():
    """S^{mu nu} = (1/4)[gamma^mu, gamma^nu] = -i sigma^{mu nu} (repo Lorentz generators)."""
    assert spin_generator_matches_lorentz_sigma()


def test_spinor_commutator_is_quarter_riemann():
    """[nabla_mu, nabla_nu] psi = +(1/4) R_{mu nu cd} gamma^c gamma^d psi, signed, curved background."""
    assert spinor_commutator_is_quarter_riemann()


def test_cartan_curvature_matches_christoffel_riemann():
    """d omega equals the frame-converted Christoffel Riemann (tetrad-postulate weld)."""
    assert cartan_curvature_matches_christoffel_riemann()


def test_flat_background_commutator_vanishes():
    """phi = const => zero connection => zero spinor curvature (flat cone limit)."""
    assert flat_background_commutator_vanishes()


def test_mutation_normalization_one_half_fails():
    """Genuine mutation: S_ab = (1/2)[gamma_a, gamma_b] injected -> comparison fails."""
    assert not mutated_normalization_one_half_matches()


def test_mutation_normalization_one_eighth_fails():
    """Genuine mutation: S_ab = (1/8)[gamma_a, gamma_b] injected -> comparison fails."""
    assert not mutated_normalization_one_eighth_matches()


def test_mutation_sign_fails():
    """Genuine mutation: opposite overall sign -> comparison fails (sign is fixed)."""
    assert not mutated_sign_matches()


def test_mutation_connection_fails():
    """Genuine mutation: non-torsion-free omega + x dy -> no longer matches the metric Riemann."""
    assert not mutated_connection_matches()
