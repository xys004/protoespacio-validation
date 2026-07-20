"""Tests for the protospace Proposition (infrared cone inheritance).

These wrap validators/protospace_theorem.py, which states and certifies a
proposition that consumes G, F, Gamma, H, L_G and G_micro -- not only the
{Gamma, H} pair that the manuscript's nullity bound uses -- together with
the sharpness objects showing those entries are load-bearing.
"""
import pytest
import sympy as sp

from validators.laplacian_projector import cycle_laplacian
from validators.protospace_theorem import (
    abelian_equivariance_forces_zero_commutator,
    chirality_descends_to_window,
    compression_identity_holds,
    cone_inheritance_bound_holds,
    cycle_cone_criterion_determines_velocity,
    cycle_hopping,
    cycle_protospace_has_exact_infrared_cone,
    decoupling_bound_is_attained,
    decoupling_bound_is_strict_on_defected_cycle,
    defected_cycle_hopping,
    defected_cycle_instance_is_nontrivial,
    expander_cone_criterion_is_velocity_blind,
    expander_window_is_trivial_below_the_gap,
    hamiltonian_level_invariants_agree,
    infrared_decoupling_bound_holds,
    laplacian_is_not_determined_by_hamiltonian,
    mutated_coordinate_window_bound_holds,
    mutated_gap_bound_holds,
    mutated_noncirculant_step_has_zero_commutator,
    mutated_ungraded_hamiltonian_keeps_window_chirality,
    nonabelian_equivariance_allows_nonzero_commutator,
    same_order_cayley_graphs_disagree_on_velocity,
    sylvester_identity_holds,
    unit_fiber_admits_no_nonzero_graded_hamiltonian,
    window_of_dimension_one_annihilates_the_laplacian,
    wrong_expander_window_is_nontrivial,
    petersen_laplacian,
)


def _instances():
    """(label, L_G, T, threshold) pairs used by the parametrized checks."""
    return [
        ("clean_c4", cycle_laplacian(4), cycle_hopping(4), 1),
        ("clean_c6", cycle_laplacian(6), cycle_hopping(6), 1),
        ("defected_c6_lam1", cycle_laplacian(6), defected_cycle_hopping(6), 1),
        ("defected_c6_lam3", cycle_laplacian(6), defected_cycle_hopping(6), 3),
    ]


INSTANCES = _instances()
IDS = [case[0] for case in INSTANCES]


# --------------------------------------------------------------------------
# The Proposition, conclusions (i)-(iii)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("case", INSTANCES, ids=IDS)
def test_sylvester_mechanism(case):
    """Q [A,H] P = A (Q H P) - (Q H P) A, the identity behind conclusion (i)."""
    _label, lap, t_block, threshold = case
    assert sylvester_identity_holds(lap, t_block, threshold)


@pytest.mark.parametrize("case", INSTANCES, ids=IDS)
def test_infrared_decoupling_bound(case):
    """Conclusion (i): ||Q H P||_F <= kappa / g_Lambda, in exact squared form."""
    _label, lap, t_block, threshold = case
    assert infrared_decoupling_bound_holds(lap, t_block, threshold)


@pytest.mark.parametrize("case", INSTANCES, ids=IDS)
def test_chirality_descends(case):
    """Conclusion (ii): the compression P H P is self-adjoint and graded."""
    _label, lap, t_block, threshold = case
    assert chirality_descends_to_window(lap, t_block, threshold)


@pytest.mark.parametrize("case", INSTANCES, ids=IDS)
def test_compression_identity(case):
    """The exact identity P H^2 P - (P H P)^2 = (Q H P)^dagger (Q H P)."""
    _label, lap, t_block, threshold = case
    assert compression_identity_holds(lap, t_block, threshold)


@pytest.mark.parametrize("case", INSTANCES, ids=IDS)
def test_cone_inheritance_bound(case):
    """Conclusion (iii): delta_IR <= delta + (kappa/g_Lambda)^2, decided exactly."""
    _label, lap, t_block, threshold = case
    assert cone_inheritance_bound_holds(lap, t_block, threshold)


# --------------------------------------------------------------------------
# The instances are neither vacuous nor slack-free
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [4, 6, 8])
def test_clean_cycle_saturates_with_zero_defect(n):
    """On C_n with T = S - I every quantity in the Proposition vanishes exactly."""
    assert cycle_protospace_has_exact_infrared_cone(n, 1)


def test_defected_instance_has_all_quantities_positive():
    """The defected cycle exercises the bounds with kappa, leakage and both
    cone defects strictly positive, so the inequalities are not vacuous."""
    assert defected_cycle_instance_is_nontrivial(6, 1)


def test_decoupling_bound_is_attained():
    """The constant 1/g_Lambda in conclusion (i) is sharp: equality on K_2."""
    assert decoupling_bound_is_attained()


def test_decoupling_bound_is_strict_elsewhere():
    """...and is a genuine inequality, strict on the defected cycle."""
    assert decoupling_bound_is_strict_on_defected_cycle(6, 1)


# --------------------------------------------------------------------------
# Corollary: G_micro
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [4, 6])
def test_abelian_equivariance_kills_the_commutator(n):
    """Any Z_n-equivariant hopping (symbolic circulant) gives [A, H] = 0."""
    assert abelian_equivariance_forces_zero_commutator(n)


def test_abelian_hypothesis_is_necessary():
    """A simply transitive NON-abelian symmetry does not force [A, H] = 0."""
    assert nonabelian_equivariance_allows_nonzero_commutator()


# --------------------------------------------------------------------------
# Sharpness: the L_G and F entries are load-bearing
# --------------------------------------------------------------------------

def test_expander_window_is_trivial():
    """Petersen (3-regular, n=10): N_Lambda = 1 for every threshold below the gap."""
    assert expander_window_is_trivial_below_the_gap()


def test_one_dimensional_window_annihilates_the_laplacian():
    """N_Lambda = 1 on a connected graph forces P_Lambda L_G P_Lambda = 0."""
    assert window_of_dimension_one_annihilates_the_laplacian(petersen_laplacian(), 1)


def test_expander_criterion_determines_no_velocity():
    """Conclusion (iv) on the expander: the in-window defect is constant in v."""
    assert expander_cone_criterion_is_velocity_blind()


def test_cycle_criterion_determines_a_velocity():
    """The contrast partner: on C_6 the defect is 4(v^2-1)^2 and pins v = 1."""
    assert cycle_cone_criterion_determines_velocity(6)


def test_same_order_cayley_graphs_disagree():
    """Two Cayley graphs on six vertices, identical in every other respect,
    give opposite verdicts on the existence of an emergent velocity."""
    assert same_order_cayley_graphs_disagree_on_velocity()


def test_laplacian_is_independent_data():
    """One (Gamma, H), two admissible L_G, opposite infrared verdicts: no
    function of (Gamma, H) alone computes the conclusion."""
    assert laplacian_is_not_determined_by_hamiltonian(6, 1)


def test_shared_hamiltonian_level_invariants():
    """The two protospaces really do share their (Gamma, H)-level data, so the
    nullity bound returns one and the same number for both."""
    assert hamiltonian_level_invariants_agree(6)


def test_unit_fiber_forbids_a_graded_hamiltonian():
    """With dim F = 1 the grading forces H = 0, so conclusion (ii) is empty."""
    assert unit_fiber_admits_no_nonzero_graded_hamiltonian()


# --------------------------------------------------------------------------
# Negative controls (genuine mutations)
# --------------------------------------------------------------------------

def test_mutated_gap_breaks_the_bound():
    """MUTATION: substituting the full spectral spread for g_Lambda falsifies (i)."""
    assert not mutated_gap_bound_holds(6, 1)


def test_mutated_coordinate_window_breaks_the_bound():
    """MUTATION: a non-spectral subspace of the same dimension falsifies (i)."""
    assert not mutated_coordinate_window_bound_holds(6, 1)


def test_mutated_ungraded_hamiltonian_breaks_chirality():
    """MUTATION: an on-site term that breaks {Gamma, H} = 0 falsifies (ii)."""
    assert not mutated_ungraded_hamiltonian_keeps_window_chirality(6, 1)


def test_mutated_noncirculant_step_breaks_the_corollary():
    """MUTATION: breaking Z_n equivariance makes kappa nonzero."""
    assert not mutated_noncirculant_step_has_zero_commutator(6)


def test_mutated_expander_window_claim_is_false():
    """MUTATION: the same counting machinery denies N_Lambda >= 2 on Petersen."""
    assert not wrong_expander_window_is_nontrivial()


# --------------------------------------------------------------------------
# Guard on the exactness discipline of the module
# --------------------------------------------------------------------------

def test_comparator_refuses_undecidable_input():
    """The exact comparator raises rather than falling back to floats, so no
    certificate in this module can silently degrade to a numerical one."""
    from validators.protospace_theorem import _decide_le

    with pytest.raises(ValueError):
        _decide_le(sp.Symbol("t"), sp.Integer(0))
