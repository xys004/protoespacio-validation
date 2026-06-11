from validators.ssh import (
    ssh_gap_closes_at_pi_when_t1_eq_t2,
    ssh_linearizes_to_dirac,
    ssh_squared_holds,
)


def test_ssh_squared():
    assert ssh_squared_holds()


def test_ssh_gap_closes():
    assert ssh_gap_closes_at_pi_when_t1_eq_t2()


def test_ssh_linearization():
    assert ssh_linearizes_to_dirac()
