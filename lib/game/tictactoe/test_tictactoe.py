import pytest
import numpy as np
from lib.game.tictactoe.tictactoe import TicTacToe


@pytest.fixture
def game():
    return TicTacToe(3, 3)


class TestConvertToMCTS:
    def test_pad(self, game):
        assert game._pad_mcts_state("120120") == "000120120"
        assert game._pad_mcts_state("0") == "000000000"

    def test_encode_game_state(self, game):
        state = [
            [1, 2, 0],
            [0, 2, 0],
            [1, 0, 0],
        ]
        assert game.encode_game_state(state) == int("120020100")

        state = [
            [0, 0, 0],
            [1, 2, 0],
            [1, 0, 0],
        ]
        assert game.encode_game_state(state) == int("000120100")

    def test_convert_mcts_state_to_list_state(self, game):
        state = [
            [0, 1, 0],
            [2, 2, 0],
            [0, 1, 1],
        ]
        assert game.convert_mcts_state_to_list_state(int("010220011")) == state


class TestPossibleMoves:
    def test_possible_moves(self, game):
        state = [
            [0, 1, 0],
            [2, 2, 0],
            [0, 1, 1],
        ]
        mcts_state = game.encode_game_state(state)
        assert game.possible_moves(mcts_state) == [3, 4]
        assert game.invalid_moves(mcts_state) == [0, 1, 2, 5, 6, 7, 8]

        state = [
            [2, 1, 2],
            [2, 2, 0],
            [0, 1, 2],
        ]
        mcts_state = game.encode_game_state(state)
        assert game.possible_moves(mcts_state) == [0, 2, 3, 4, 8]
        assert game.invalid_moves(mcts_state) == [1, 5, 6, 7]


class TestStatesToTrainingBatch:
    def test_states_to_training_batch(self, game):
        states = [int('001010221'), int('101222001')]
        who_moves = [1, 0]
        batch = game.states_to_training_batch(states, who_moves)
        batch_state_1 = [
            # from player 1's POV
            [
                # player 1's tokens (self)
                [0, 0, 1],
                [0, 1, 0],
                [0, 0, 1]
            ],
            [
                # player 0's tokens (opponent)
                [1, 1, 0],
                [1, 0, 1],
                [0, 0, 0]
            ]
        ]

        batch_state_2 = [
            # from player 0's POV
            [
                # player 0's tokens (self)
                [0, 1, 0],
                [0, 0, 0],
                [1, 1, 0]
            ],
            [
                # player 1's tokens (opponent)
                [1, 0, 1],
                [0, 0, 0],
                [0, 0, 1]
            ]
        ]

        np.testing.assert_equal(batch, [batch_state_1, batch_state_2])


class TestMove:
    def test_moves(game):
        board = int('222222222')
        won, board = game.move(board, 0, 1)
        assert won == False
        assert board == int('202222222')

        won, board = game.move(board, 1, 5)
        assert won == False
        assert board == int('202221222')

        won, board = game.move(board, 0, 8)
        assert won == False
        assert board == int('202222220')

        won, board = game.move(board, 1, 7)
        assert won == False
        assert board == int('202222210')

    def test_winning_moves(game):
        board = int('002112122')
        won, new_board = game.move(board, 0, 2)
        assert won == True
        assert new_board == int('000112122')

        board = int('021012212')
        won, new_board = game.move(board, 0, 6)
        assert won == True
        assert new_board == int('021012012')

        board = int('021102212')
        won, new_board = game.move(board, 0, 9)
        assert won == True
        assert new_board == int('021102210')

        board = int('120122012')
        won, new_board = game.move(board, 0, 4)
        assert won == True
        assert new_board == int('120102012')
