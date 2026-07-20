from validators.causal_order import (
    bloch_form_matches_real_space_walk,
    coin_grading_leaves_causal_order_flat,
    graded_coin_differs_from_uniform_in_operator_norm,
    continuum_sprinkling_recovers_cone_slope,
    continuum_sprinkling_recovers_cone_slope_3plus1,
    dimension_estimator_rejects_total_order,
    dimension_estimator_separates_3plus1,
    emergent_cone_inside_strict_cone,
    flipflop_walk_has_no_reconstructible_cone,
    graded_lattice_variable_cone_recovered_locally,
    influence_support_tolerance_has_margin,
    local_walk_passes_reconstruction_pipeline,
    massless_order_is_transitive_on_a_single_causal_cone,
    massless_substrate_causal_set_degenerates_to_two_chains,
    massless_support_relation_needs_transitive_closure,
    mutated_event_density_fails_proper_time,
    mutated_wormhole_walk_breaks_substrate_reconstruction,
    nonlocal_step_has_no_cone,
    recovered_cone_is_the_strict_cone_not_the_infrared_cone,
    reconstruction_tracks_substrate_shift_range,
    substrate_cone_reconstruction_converges_with_lattice_size,
    substrate_counting_recovers_proper_time,
    substrate_counting_tracks_lorentzian_not_coordinate_separation,
    substrate_order_is_strict_partial_order,
    substrate_order_myrheim_meyer_dimension_is_two,
    substrate_order_recovers_strict_cone_slope,
    substrate_order_splits_into_two_parity_components,
    support_cone_edge_is_attained,
    support_outside_ball_is_exactly_zero,
    wrong_infrared_speed_fails_substrate_bracket,
    wrong_uniform_embedding_shows_no_cone_grading,
)


# --- the substrate step and its support cone --------------------------------

def test_support_outside_ball_is_exactly_zero():
    """Locality of the substrate step: exact zero amplitude outside the ball."""
    assert support_outside_ball_is_exactly_zero()


def test_support_cone_edge_is_attained():
    """The strict support cone has slope exactly 1 site/step (edge attained)."""
    assert support_cone_edge_is_attained()


def test_influence_support_tolerance_has_margin():
    """The 1e-12 support threshold is audited: smallest nonzero block > 1e-7."""
    assert influence_support_tolerance_has_margin()


# --- the substrate causal order ---------------------------------------------

def test_substrate_order_is_strict_partial_order():
    """The walk's own influence relation is a strict partial order for 0 < theta < pi/2."""
    assert substrate_order_is_strict_partial_order()


def test_massless_support_relation_needs_transitive_closure():
    """At theta = 0 the raw relation is not transitive; its closure IS the massive order."""
    assert massless_support_relation_needs_transitive_closure()


def test_massless_order_is_transitive_on_a_single_causal_cone():
    """Restricted to one origin's causal future, the massless relation is a partial order."""
    assert massless_order_is_transitive_on_a_single_causal_cone()


def test_substrate_order_splits_into_two_parity_components():
    """The coined walk's order is bipartite: two components, and no whole-lattice bracket."""
    assert substrate_order_splits_into_two_parity_components()


# --- Malament / HKM executed on the substrate order -------------------------

def test_substrate_order_recovers_strict_cone_slope():
    """Bracketing the walk's OWN order returns the walk's own strict support speed."""
    assert substrate_order_recovers_strict_cone_slope()


def test_substrate_cone_reconstruction_converges_with_lattice_size():
    """The bracket width is 2c/(N-1): the substrate reconstruction converges as O(1/N)."""
    assert substrate_cone_reconstruction_converges_with_lattice_size()


def test_recovered_cone_is_the_strict_cone_not_the_infrared_cone():
    """The order returns the lattice cone (slope 1), mass independent, not cos(theta)."""
    assert recovered_cone_is_the_strict_cone_not_the_infrared_cone()


def test_reconstruction_tracks_substrate_shift_range():
    """Change the substrate (shift range s = 1, 2, 3) and the recovered slope becomes s."""
    assert reconstruction_tracks_substrate_shift_range()


def test_local_walk_passes_reconstruction_pipeline():
    """Positive anchor: the unmodified local walk passes the pipeline the mutations use."""
    assert local_walk_passes_reconstruction_pipeline()


# --- volume counting on the substrate order ---------------------------------

def test_substrate_counting_recovers_proper_time():
    """Counting the walk's own Alexandrov interval recovers tau, with relerr*N(1-k^2) -> 2."""
    assert substrate_counting_recovers_proper_time()


def test_substrate_counting_tracks_lorentzian_not_coordinate_separation():
    """The count follows sqrt(N^2 - X^2), not the common coordinate time N."""
    assert substrate_counting_tracks_lorentzian_not_coordinate_separation()


def test_substrate_order_myrheim_meyer_dimension_is_two():
    """Ordering fraction inside a substrate interval -> 1/2 as 1/2 + 1/(a+1)."""
    assert substrate_order_myrheim_meyer_dimension_is_two()


def test_massless_substrate_causal_set_degenerates_to_two_chains():
    """Mass is what makes the substrate causal set 2D; the volume rung needs theta > 0."""
    assert massless_substrate_causal_set_degenerates_to_two_chains()


# --- where the position dependence of the cone comes from -------------------

def test_coin_grading_leaves_causal_order_flat():
    """A position-dependent coin gives a bit-identical causal order: the order ignores mass."""
    assert coin_grading_leaves_causal_order_flat()


def test_graded_coin_is_a_large_deformation():
    """The coin grading differs from the uniform walk by 0.2493 in operator norm."""
    assert graded_coin_differs_from_uniform_in_operator_norm()


def test_graded_lattice_variable_cone_recovered_locally():
    """On a graded chain the local cone c(X) = local vertex spacing is read off the order."""
    assert graded_lattice_variable_cone_recovered_locally()


# --- the infrared cone, from the step operator ------------------------------

def test_bloch_form_matches_real_space_walk():
    """The Bloch step operator is the real-space walk, checked on every allowed k."""
    assert bloch_form_matches_real_space_walk()


def test_emergent_cone_inside_strict_cone():
    """IR metric cone (v = cos theta) inside the lattice cone; equal iff massless."""
    assert emergent_cone_inside_strict_cone()


# --- continuum-side checks, explicitly NOT substrate results ----------------

def test_continuum_sprinkling_recovers_cone_slope():
    """Continuum control only: the bracketing algorithm is correct on a 1+1 sprinkling."""
    assert continuum_sprinkling_recovers_cone_slope()


def test_continuum_sprinkling_recovers_cone_slope_3plus1():
    """Continuum control only: one isotropic SO(3) cone, the substrate here being 1+1."""
    assert continuum_sprinkling_recovers_cone_slope_3plus1()


# --- negative controls and genuine mutations --------------------------------

def test_flipflop_walk_has_no_reconstructible_cone():
    """At theta = pi/2 there is no partial order and no bracket: no cone is manufactured."""
    assert flipflop_walk_has_no_reconstructible_cone()


def test_wrong_infrared_speed_fails_substrate_bracket():
    """MUTATION: certifying against cos(theta) instead of the strict speed must fail."""
    assert not wrong_infrared_speed_fails_substrate_bracket()


def test_mutated_wormhole_walk_breaks_substrate_reconstruction():
    """MUTATION: a still-unitary but nonlocal shift breaks the same reconstruction pipeline."""
    assert not mutated_wormhole_walk_breaks_substrate_reconstruction()


def test_wrong_uniform_embedding_shows_no_cone_grading():
    """MUTATION: with a uniform embedding the local estimator reports no grading."""
    assert not wrong_uniform_embedding_shows_no_cone_grading()


def test_mutated_event_density_fails_proper_time():
    """MUTATION: the naive density rho = 1 instead of the substrate's 1/2 spoils tau."""
    assert not mutated_event_density_fails_proper_time()


def test_dimension_estimator_rejects_total_order():
    """Chain control: ordering fraction 1, not manifold-like."""
    assert dimension_estimator_rejects_total_order()


def test_dimension_estimator_separates_3plus1():
    """3+1 sprinkling control: ordering fraction well below the 1+1 value."""
    assert dimension_estimator_separates_3plus1()


def test_nonlocal_step_has_no_cone():
    """Wormhole-edge control: ball growth violates the cone bound; local walk saturates it."""
    assert nonlocal_step_has_no_cone()
