import numpy as np
import pytest

from count_islands.count_islands import count_islands, ERROR_2D_ARRAY, ERROR_ZEROS_AND_ONES

NESTED_ISLANDS = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
]

SNAKE_ISLANDS = [
    [1, 0, 0, 1, 1],
    [0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0],
    [1, 1, 0, 0, 1],
]

LONG_ISLANDS = [[1, 0, 1, 0, 1]] * 100

WIDE_ISLANDS = [
    [1] * 100,
    [0] * 100,
    [1] * 100,
    [0] * 100,
    [1] * 100,
]

ONE_COLUMN_ISLANDS = [[1], [0]] * 100
ONE_ROW_ISLANDS = [[1, 0] * 100]
TEN_THOUSAND_ISLANDS = [[1, 0] * 100, [0] * 200] * 100


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (np.array([]), 0),
        (np.zeros((100, 100)), 0),
        (np.ones((100, 100)), 1),
        (np.eye(100), 1),
        (np.array(NESTED_ISLANDS), 2),
        (np.array(SNAKE_ISLANDS), 3),
        (np.array(LONG_ISLANDS), 3),
        (np.array(WIDE_ISLANDS), 3),
        (np.array(ONE_COLUMN_ISLANDS), 100),
        (np.array(ONE_ROW_ISLANDS), 100),
        (np.array(TEN_THOUSAND_ISLANDS), 10**4),
    ],
)
def test_count_islands(test_input, expected):
    assert count_islands(test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected_msg",
    [
        (np.ones((3,)), ERROR_2D_ARRAY),
        (np.ones((3, 3, 3)), ERROR_2D_ARRAY),
        (np.array([["a", "b"], ["c", "d"]]), ERROR_ZEROS_AND_ONES),
        (np.array([["1", "1"], ["0", "0"]]), ERROR_ZEROS_AND_ONES),
        (np.array([[1, 2], [3, 4]]), ERROR_ZEROS_AND_ONES),
    ],
)
def test_count_islands_fails(test_input, expected_msg):
    with pytest.raises(ValueError) as e_info:
        count_islands(test_input)
    assert e_info.value.args[0] == expected_msg
