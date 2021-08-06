import re
import numpy as np
from lib.game.game import BaseGame
from lib.game.tictactoe import tictactoe_helpers
from typing import List, Tuple

Matrix = List[List[int]]


class TicTacToe(BaseGame):
    """A representation of a subset of the general m,n,k game
    (https://en.wikipedia.org/wiki/M,n,k-game)
    with n=m and winning is by simply placing k tokens adjacent
    to one another without any further conditions.
    The game state can be represented in 2 ways:
    - As a nxn list of lists of ints
    where 1 & 0 represent tokens places by either player, and 2 represents
    an empty square, a.k.a "Matrix" form
    - As an nxn-digit integer, with the same meaning for 0, 1, and 2. The position
    of each digit corresponding to the index of each square on the board from
    top to bottom and left to right, e.g. on a 3x3 board:
    |0|1|2|
    |3|4|5|
    |6|7|8|
    """

    def __init__(self, n: int = 3, k_to_win: int = 3):
        """Create an instance of the game.
        With default arguments the game is a Tic Tac Toe game.

        Args:
            n (int, optional): Number of squares for each side of the game board.
                Defaults to 3.
            k_to_win (int, optional): Number of consecutive tokens to win.
                Defaults to 3.
        """
        super().__init__()
        self.board_len = n
        self.k_to_win = k_to_win
        self.player_black = 1
        self.player_white = 0
        self.empty = 2

    @staticmethod
    def flatten_nested_list(nested_list: List[List]) -> List:
        """Unravel a double-nested list into a flat list

        Args:
            nested_list (List[List])

        Returns:
            List
        """
        return [item for sublist in nested_list for item in sublist]

    @property
    def initial_state(self) -> int:
        """The initial state of the game in MCTS form. This is used in
        utils.play_game to start the MCTS loop.
        """
        empty_board = np.full(
            (self.board_len, self.board_len), self.empty).tolist()
        return self.encode_game_state(empty_board)

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        """The shape of the neural network form of the game state.
        This should be a tuple of ints. e.g. For a game of Tic-Tac-Toe it's
        likely (2, 3, 3) i.e. the game state being fed into the neural net
        is a 2x3x3 tensor: 2 player x 3x3 board (board viewed from each player's
        side)
        """
        return (2, self.board_len, self.board_len)

    @property
    def action_space(self) -> int:
        """The total count of all possible actions that can be performed.
        This is a constant for each game that represents all actions,
        regardless of whether they are valid at each game state.

        Used to initialize MCTS nodes (determine how large the values, avg_values &
        visit count vectors are)

        Returns:
            (int): count of total possible actions
        """
        return self.board_len ** 2

    def _pad_mcts_state(self, mcts_state_str: str) -> str:
        """Since the game state in int form might have leading zeroes,
        We have to pad it to proper length before converting to Matrix form

        Args:
            mcts_state (int): Game state as int

        Returns:
            str: String of game state in MCTS-friendly form, padded
            to appropriate length with leading zeroes
        """
        return mcts_state_str.rjust(self.board_len ** 2, "0")

    def encode_game_state(self, state_list: Matrix) -> int:
        """Convert game state from Matrix form to MCTS-friendly form (int)
        which is smaller to store and hashable for quick looking in MCTS search

        Args:
            state_list (Matrix): Game state as list of lists of tokens

        Returns:
            int: Game state as int
        """
        flattened = self.flatten_nested_list(state_list)
        stringified = [str(i) for i in flattened]
        return int(''.join(stringified))

    def convert_mcts_state_to_list_state(self, mcts_state: int) -> Matrix:
        """Convert game state from more compact, hashable i.e. MCTS-friendly form
        to a form that is easier for us to think about (matrix of tokens)

        Args:
            mcts_state (int): Game state in MCTS-friendly form

        Returns:
            (Matrix): Game state in more human-friendly form (list of list of tokens)
        """
        # Pad to number of squares on board (in case of leading zeros)
        padded = self._pad_mcts_state(str(mcts_state))
        state = []
        for i, c in enumerate(padded):
            if i % self.board_len == 0:
                # new row only every board_len items
                state.append([int(c)])
            else:
                state[i // self.board_len].append(int(c))
        return state

    def possible_moves(self, mcts_state: int) -> List:
        """Return indexes of empty squares, left to right, top to bottom
        |0|1|2|
        |3|4|5|
        |6|7|8|

        Args:
            mcts_state (int): Game state in MCTS-friendly form

        Returns:
            Iterable: [description]
        """
        padded = self._pad_mcts_state(str(mcts_state))
        return [i for i, c in enumerate(padded) if c == str(self.empty)]

    def invalid_moves(self, mcts_state: int) -> List:
        """Return the non-empty squares

        Args:
            mcts_state (int): Game state in MCTS-friendly form

        Returns:
            List: List of illegal moves (occupied squares)
        """
        padded = self._pad_mcts_state(str(mcts_state))
        return [i for i, c in enumerate(padded) if c != str(self.empty)]

    def _encode_list_state(self, dest_np: np.ndarray, state: Matrix, who_move: int) -> None:
        """
        In-place encodes list state into the zero numpy array
        :param dest_np: dest array, expected to be zeroes
        :param who_move: player index (game.player_white or game.player_black)
        """
        assert dest_np.shape == self.obs_shape

        for row_idx, row in enumerate(state):
            for col_idx, cell in enumerate(row):
                if cell == who_move:
                    dest_np[0, row_idx, col_idx] = 1.0
                elif cell != self.empty:
                    dest_np[1, row_idx, col_idx] = 1.0

    def states_to_training_batch(self, state_ints: List[int],
                                 who_moves_lists: List[int]) -> np.ndarray:
        """Convert game states to arrays that can be fed to neural network

        Args:
            state_ints (List[int]): List of game states in MCTS form
            who_moves_lists (List[int]): Corresponding list of player whose move
                led to the game state

        Returns:
            np.array:  each game state will be represented as a
            2 x board_len x board_len array. Each array will reflect the
            tokens/moves of one player with value 1 and have zero values in
            all other locations (opposing player's token and empty locations).
            This is how the data is fed to the neural network in the AlphaZero paper.
            e.g.:
            |0|1|
            | |0| with player 0 becomes:

            [[[1, 0],
              [0, 1]],
             [[0, 1],
              [0, 0]]]
        """
        batch_size = len(state_ints)
        batch = np.zeros((batch_size,) + self.obs_shape, dtype=np.float32)
        for idx, (state, who_move) in enumerate(zip(state_ints, who_moves_lists)):
            converted_state = self.convert_mcts_state_to_list_state(state)
            self._encode_list_state(batch[idx], converted_state, who_move)
        return batch

    def move(self, mcts_state: int, move: int, player: int) -> Tuple[int, bool]:
        """At a given game state, make a (legal) move by a specified player

        Args:
            mcts_state (int): Game state in MCTS form
            move (int): Location of the move, can be calculated as index of square
                on the board from top to bottom, right to left e.g.
                |0|1|2|
                |3|4|5|
                |6|7|8|
            player (int): 0 or 1, which player is making the move

        Returns:
            Tuple[int, bool]: New game state & if the game had been won by the
            player who just made the move
        """
        assert player == self.player_white or player == self.player_black
        assert move >= 0 and move <= self.action_space

        board = self.convert_mcts_state_to_list_state(mcts_state)
        row_idx, col_idx = divmod(move, self.board_len)
        board[row_idx][col_idx] = player
        won = tictactoe_helpers.check_win(
            board, (row_idx, col_idx), self.k_to_win, player)
        new_mcts_state = self.encode_game_state(board)
        return new_mcts_state, won

    def render(self, mcts_state: int) -> str:
        """String representation of board, for interacting with human player

        Args:
            mcts_state (int): Game state in MCTS form

        Returns:
            str: String representation of game state
        """
        list_state = self.convert_mcts_state_to_list_state(mcts_state)
        for row_idx, row in enumerate(list_state):
            for col_idx, cell in enumerate(row):
                if cell == self.empty:
                    list_state[row_idx][col_idx] = str(
                        row_idx * self.board_len + col_idx)
                elif cell == self.player_white:
                    list_state[row_idx][col_idx] = ❌
                elif cell == self.player_black:
                    list_state[row_idx][col_idx] = ⭕
        # substitute semi-colons with pipe |
        list_str = [f'|{"|".join(row)}|' for row in list_state]
        board = '\n'.join(list_str).replace(',', '')
        return board
