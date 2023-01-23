import collections
import itertools
from argparse import ArgumentParser, FileType
from io import TextIOWrapper
from typing import List, Tuple

import numpy as np

OFFSETS = [offset for offset in itertools.product([-1, 0, 1], repeat=2) if any(offset)]
ERROR_2D_ARRAY = "Invalid data provided. Input data should be a two-dimensional numpy array."
ERROR_ZEROS_AND_ONES = "Invalid data provided. Input data should be a numpy array consisting of zeroes and ones."


def validate_data(data: np.ndarray):
    if not len(data.shape) == 2:
        raise ValueError(ERROR_2D_ARRAY)
    if not np.isin(data, (0, 1)).all():
        raise ValueError(ERROR_ZEROS_AND_ONES)


def get_unvisited_neighbors(idx: Tuple[int, int], to_visit: np.ndarray) -> List[Tuple[int, int]]:
    neighbors = []
    for offset in OFFSETS:
        neighbor = idx[0] + offset[0], idx[1] + offset[1]
        try:
            if to_visit[neighbor] == 1 and neighbor[0] >= 0 and neighbor[1] >= 0:
                neighbors.append(neighbor)
        except IndexError:
            continue
    return neighbors


def visit_island(idx: Tuple[int, int], to_visit: np.ndarray, remaining: int) -> int:
    queue = collections.deque([idx])
    to_visit[idx] = 0
    while queue:
        next_idx = queue.popleft()
        remaining -= 1
        for neighbor in get_unvisited_neighbors(next_idx, to_visit):
            to_visit[neighbor] = 0
            queue.append(neighbor)
    return remaining


def get_next_to_visit(to_visit: np.ndarray, idx: int) -> int:
    to_visit_size = to_visit.size
    to_visit_flat = to_visit.flat
    while idx < to_visit_size:
        if to_visit_flat[idx] == 1:
            return idx
        idx += 1


def count_islands(data: np.ndarray) -> int:
    """
    Counts the number of islands, denoted as groups of adjacent, numeric ones
    (integer or float) in a two-dimensional numpy array consisting of ones
    and zeros.

    Notes
    -----
    Two positions in an array are considered adjacent if the offset between
    them in the array is no greater than one on any axis, i.e. position
    with coordinates (0, 0) is adjacent to positions (0, 1), (1, 1) and (1, 0).

    Parameters:
    -----
    data : np.ndarray
        The array to count islands in.

    Returns:
    -----
    count : int
        Count of the islands in the `data` array.
    """
    count = 0
    if data.size:
        validate_data(data)
        to_visit = data.copy()
        remaining = np.sum(to_visit)
        idx_flat = 0
        while remaining:
            count += 1
            idx_flat = get_next_to_visit(to_visit, idx_flat)
            idx = np.unravel_index(idx_flat, to_visit.shape)
            remaining = visit_island(idx, to_visit, remaining)
    return count


def count_islands_in_file(file_obj: TextIOWrapper) -> int:
    return count_islands(np.genfromtxt(file_obj, delimiter=1))


def main():
    description = """
    Reads in the two-dimensional array of ones and zeros from the text file under `filename` and prints the number
    of islands consisting of groups of adjacent ones to the sys.stdout.
    """
    parser = ArgumentParser(description=description)
    parser.add_argument("filename", help="The name of the file containing input data.", type=FileType("r"))
    args = parser.parse_args()
    print(count_islands_in_file(args.filename))


if __name__ == "__main__":
    main()
