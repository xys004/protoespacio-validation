from validators.causal_order import (
    causal_relation_is_partial_order,
    counting_recovers_proper_time,
    dimension_estimator_rejects_total_order,
    dimension_estimator_separates_3plus1,
    emergent_cone_inside_strict_cone,
    myrheim_meyer_dimension_is_two,
    nonlocal_step_has_no_cone,
    order_recovers_light_cone_slope,
    order_recovers_light_cone_slope_3plus1,
    support_cone_edge_is_attained,
    support_outside_ball_is_exactly_zero,
)


def test_support_outside_ball_is_exactly_zero():
    """Locality of the substrate step: exact zero amplitude outside the ball."""
    assert support_outside_ball_is_exactly_zero()


def test_support_cone_edge_is_attained():
    """The strict support cone has slope exactly 1 site/step (edge attained)."""
    assert support_cone_edge_is_attained()


def test_causal_relation_is_partial_order():
    """The influence relation on events is a strict causal partial order."""
    assert causal_relation_is_partial_order()


def test_emergent_cone_inside_strict_cone():
    """IR metric cone (v = cos theta) inside the lattice cone; equal iff massless."""
    assert emergent_cone_inside_strict_cone()


def test_order_recovers_light_cone_slope():
    """Malament/HKM executable: order alone brackets the cone slope tightly."""
    assert order_recovers_light_cone_slope()


def test_order_recovers_light_cone_slope_3plus1():
    """The full (3+1) cone (not a 1+1 toy) is fixed by the order alone."""
    assert order_recovers_light_cone_slope_3plus1()


def test_counting_recovers_proper_time():
    """Causal-set volume: interval counting recovers proper time (full metric)."""
    assert counting_recovers_proper_time()


def test_variable_cone_recovered_locally():
    """Position-dependent cone c(x) re-extracted locally from pure order data."""
    from validators.causal_order import variable_cone_recovered_locally

    assert variable_cone_recovered_locally()


def test_myrheim_meyer_dimension_is_two():
    """Ordering fraction 1/2 in the 1+1 diamond: the order knows the dimension."""
    assert myrheim_meyer_dimension_is_two()


def test_dimension_estimator_rejects_total_order():
    """Chain control: ordering fraction 1, not manifold-like."""
    assert dimension_estimator_rejects_total_order()


def test_dimension_estimator_separates_3plus1():
    """3+1 sprinkling control: ordering fraction well below the 1+1 value."""
    assert dimension_estimator_separates_3plus1()


def test_nonlocal_step_has_no_cone():
    """Wormhole-edge control: ball growth violates the cone bound; local walk saturates it."""
    assert nonlocal_step_has_no_cone()
