from typing import List, Tuple

Matrix = List[List[int]]
Coord = Tuple[int, int]


def check_win(matrix: Matrix, move: Coord, k: int, token: int) -> bool:
    """[summary]

    Args:
        matrix (Matrix): [description]
        move (Coord): [description]
        k (int): [description]
        token (int): [description]

    Returns:
        bool: [description]
    """
    for func in (get_row, get_col, get_diag, get_antidiag):
        arr = func(matrix, move)
        if k_in_a_row(arr, k, token):
            return True
    return False


def k_in_a_row(arr: List[int], k: int, token: int) -> bool:
    """[summary]

    Args:
        arr ([type]): [description]
        k ([type]): [description]
        token ([type]): [description]
    """
    assert k > 1, "We do not handle trivial cases where k <= 1"
    if len(arr) < k:
        # Impossible
        return False

    matchStartIndex = None
    for i in range(len(arr)):
        if arr[i] == token:
            if matchStartIndex is None:
                matchStartIndex = i
            elif (i - matchStartIndex + 1) >= k:
                # There are >= k tokens, we can return
                return True
        else:
            # A non-matching token found. Reset
            matchStartIndex = None
            if i >= (len(arr) - k):
                # There are less than k tokens left, we cannot match
                return False


def get_row(matrix: Matrix, coord: Coord) -> List[int]:
    """[summary]

    Args:
        matrix (Matrix): [description]
        coord (Coord): [description]

    Returns:
        [type]: [description]
    """
    return matrix[coord[0]]


def get_col(matrix: Matrix, coord: Coord) -> List[int]:
    """[summary]

    Args:
        matrix (Matrix): [description]
        coord (Coord): [description]
    """
    col_idx = coord[1]
    return [row[col_idx] for row in matrix]


def get_diag(matrix: Matrix, coord: Coord) -> List[int]:
    """[summary]

    _|0|1|2|3|
    0|_|_|_|_|
    1|_|_|_|_|
    2|_|_|_|_|
    3|_|_|_|_|

    Args:
        matrix (Matrix): [description]
        coord (Coord): [description]

    Returns:
        List[int]: [description]
    """
    row_idx = coord[0]
    col_idx = coord[1]

    if row_idx >= col_idx:
        # These values are true for the main diagonal
        # (top-left to bottom-right through board center)
        # and other smaller diagonals below it
        # These diagonals always contain the first col
        col_start = 0
        row_start = row_idx - col_idx
        # These diagonals always contain the last row
        row_end = len(matrix) - 1
        # Due to symmetry.
        # Draw and observe the row where the diagonal ends
        # is the inverted index of the column where it begins
        col_end = row_end - col_start
    else:
        # Diagonals above the main always contain the first row
        row_start = 0
        col_start = col_idx - row_idx
        # and the last column
        col_end = len(matrix) - 1
        row_end = col_end - col_start

    diag = []
    x = row_start
    y = col_start
    while x <= row_end and y <= col_end:
        diag.append(matrix[x][y])
        x += 1
        y += 1
    return diag


def get_antidiag(matrix: Matrix, coord: Coord) -> List[int]:
    """We get the anti-diagonals (bottom-left to top-right)

    Args:
        matrix (Matrix): [description]
        coord (Coord): [description]

    Returns:
        List[int]: [description]
    """
    assert len(matrix) == len(matrix[0]), "we only handle squares"
    row_idx = coord[0]
    col_idx = coord[1]

    if row_idx + col_idx < len(matrix):
        # These values are true for the main anti-diagonal
        # (bottom-left to top-right through board center)
        # and other smaller diagonals above it
        # These diagonals always contain the first col
        col_start = 0
        # bottom-most row. We start decrementing from here
        row_start = row_idx + col_idx
        # These anti-diagonals always contain the last row
        row_end = 0
        # Due to symmetry.
        # Draw and observe the column where the anti-diagonal ends
        # is same index as the row where it begins
        col_end = row_start
    else:
        # Anti-diagonals below the main always contain the last column
        col_end = len(matrix) - 1
        col_start = col_idx + row_idx - col_end
        # And the last row. We also start decrementing from here
        row_start = len(matrix) - 1
        row_end = col_start

    anti = []
    x = row_start
    y = col_start
    while x >= row_end and y <= col_end:
        anti.append(matrix[x][y])
        # Collecting from bottom of matrix to top
        x -= 1
        # We are still collecting from left to right
        # so we still increment y
        y += 1
    return anti
