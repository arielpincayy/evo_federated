import pytest

from evofederated.experiments.runner_federated import _limit_participants


def test_pair_schedule_keeps_one_representative_per_pair():
    assert _limit_participants([6, 2, 7, 5], 4, range(8)) == [2, 5, 6, 7]


def test_pair_schedule_rejects_arbitrary_truncation():
    with pytest.raises(ValueError, match="one representative per pair"):
        _limit_participants([0, 1, 2, 3], 2, range(8))
