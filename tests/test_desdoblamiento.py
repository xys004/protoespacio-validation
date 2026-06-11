from validators.desdoblamiento import (
    lower_weyl_node_at_plus_b_over_v,
    nodes_separated_by_two_b_over_v,
    squared_decomposition_holds,
    upper_weyl_node_at_minus_b_over_v,
)


def test_squared_decomposition():
    assert squared_decomposition_holds()


def test_upper_weyl_node():
    assert upper_weyl_node_at_minus_b_over_v()


def test_lower_weyl_node():
    assert lower_weyl_node_at_plus_b_over_v()


def test_node_separation():
    assert nodes_separated_by_two_b_over_v()
