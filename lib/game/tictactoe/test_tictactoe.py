import pytest
from lib.game.tictactoe import tictactoe_helpers


class TestGetters:
    @pytest.fixture
    def tictactoe_board(self):
        return [
            [1, -1, 1],
            [-1, -1, 0],
            [0, -1, 1]
        ]

    def test_get_col(self, tictactoe_board):
        assert tictactoe_helpers.get_col(tictactoe_board, [0, 0]) == [1, -1, 0]
        assert tictactoe_helpers.get_col(tictactoe_board, [1, 0]) == [1, -1, 0]
        assert tictactoe_helpers.get_col(
            tictactoe_board, [2, 1]) == [-1, -1, -1]
        assert tictactoe_helpers.get_col(tictactoe_board, [1, 2]) == [1, 0, 1]

    def test_get_diag(self, tictactoe_board):
        assert tictactoe_helpers.get_diag(
            tictactoe_board, [0, 0]) == [1, -1, 1]
        assert tictactoe_helpers.get_diag(tictactoe_board, [1, 0]) == [-1, -1]
        assert tictactoe_helpers.get_diag(tictactoe_board, [2, 1]) == [-1, -1]
        assert tictactoe_helpers.get_diag(tictactoe_board, [1, 2]) == [-1, 0]
        assert tictactoe_helpers.get_diag(
            tictactoe_board, [1, 1]) == [1, -1, 1]

    def test_get_antidiag(self, tictactoe_board):
        assert tictactoe_helpers.get_antidiag(tictactoe_board, [0, 0]) == [1]
        assert tictactoe_helpers.get_antidiag(
            tictactoe_board, [1, 0]) == [-1, -1]
        assert tictactoe_helpers.get_antidiag(
            tictactoe_board, [2, 1]) == [-1, 0]
        assert tictactoe_helpers.get_antidiag(
            tictactoe_board, [1, 2]) == [-1, 0]
        assert tictactoe_helpers.get_antidiag(
            tictactoe_board, [1, 1]) == [0, -1, 1]


class TestKInARow:
    def test_short_arr(self):
        assert tictactoe_helpers.k_in_a_row([1, 1, 1], 3, 1) == True
        assert tictactoe_helpers.k_in_a_row([-1, -1, -1], 3, -1) == True
        assert tictactoe_helpers.k_in_a_row([1, 0, 1], 3, 1) == False
        assert tictactoe_helpers.k_in_a_row([-1, -1, 1], 3, -1) == False

    def test_long_arr(self):
        assert tictactoe_helpers.k_in_a_row([1, 1, 1, 0], 3, 1) == True
        assert tictactoe_helpers.k_in_a_row([0, -1, -1, -1], 3, -1) == True
        assert tictactoe_helpers.k_in_a_row([1, 0, 1, 1], 3, 1) == False
        assert tictactoe_helpers.k_in_a_row([-1, 1, -1, 1], 3, -1) == False


class TestCheckWin:
    def check_col_win(self):
        board = [
            [1, -1, 1],
            [0, -1, 0],
            [0, -1, 1]
        ]
        assert tictactoe_helpers.check_win(board, 3, -1) == True
        assert tictactoe_helpers.check_win(board, 3, 1) == False

    def check_row_win(self):
        board = [
            [1, 1, 1],
            [0, -1, 0],
            [0, -1, 1]
        ]
        assert tictactoe_helpers.check_win(board, 3, 1) == True
        assert tictactoe_helpers.check_win(board, 3, -1) == False

    def check_diag_win(self):
        board = [
            [1, -1, 1],
            [0, 1, 0],
            [0, -1, 1]
        ]
        assert tictactoe_helpers.check_win(board, 3, 1) == True
        assert tictactoe_helpers.check_win(board, 3, -1) == False

    def check_antidiag_win(self):
        board = [
            [0, -1, 1],
            [0, 1, 0],
            [1, -1, 1]
        ]
        assert tictactoe_helpers.check_win(board, 3, 1) == True
        assert tictactoe_helpers.check_win(board, 3, -1) == False

    def no_win(self):
        board = [
            [0, -1, 1],
            [0, 0, 0],
            [1, -1, 1]
        ]
        assert tictactoe_helpers.check_win(board, 3, 1) == False
        assert tictactoe_helpers.check_win(board, 3, -1) == False
