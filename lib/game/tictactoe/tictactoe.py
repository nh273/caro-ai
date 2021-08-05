import numpy as np
from lib.game.game import BaseGame
from lib.game.tictactoe import tictactoe_helpers
from typing import List, Tuple

Matrix = List[List[int]]


class TicTacToe(BaseGame):
    """Represents the Tic
    """

    def __init__(self, n: int = 3, k_to_win: int = 3):
        """[summary]

        Args:
            n (int, optional): [description]. Defaults to 3.
            k_to_win (int, optional): [description]. Defaults to 3.
        """
        super().__init__()
        self.board_len = n
        self.k_to_win = k_to_win
        self.player_black = 1
        self.player_white = 0
        self.empty = 2

    @staticmethod
    def flatten_nested_list(nested_list: List[List]) -> List:
        """[summary]

        Args:
            nested_list (List[List]): [description]

        Returns:
            List: [description]
        """
        return [item for sublist in nested_list for item in sublist]

    @property
    def initial_state(self) -> int:
        """The initial state of the game in MCTS form. This is used in
        utils.play_game to start the MCTS loop.
        """
        empty_board = np.full((self.board_len, self.board_len), self.empty)
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
        """[summary]

        Args:
            mcts_state (int): [description]

        Returns:
            str: [description]
        """
        return mcts_state_str.rjust(self.board_len ** 2, "0")

    def encode_game_state(self, state_list: List[List[int]]) -> int:
        """[summary]

        Args:
            state_list (List[List[int]]): [description]

        Returns:
            int: [description]
        """
        flattened = self.flatten_nested_list(state_list)
        stringified = [str(i) for i in flattened]
        return int(''.join(stringified))

    def convert_mcts_state_to_list_state(self, mcts_state: int) -> Matrix:
        """[summary]

        Args:
            mcts_state (Hashable): [description]

        Returns:
            (Matrix): [description]
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
        """[summary]

        Args:
            mcts_state (Hashable): [description]

        Returns:
            Iterable: [description]
        """
        padded = self._pad_mcts_state(str(mcts_state))
        return [i for i, c in enumerate(padded) if c == str(self.empty)]

    def invalid_moves(self, mcts_state: int) -> List:
        """[summary]

        Args:
            mcts_state (Hashable): [description]

        Returns:
            List: [description]
        """
        padded = self._pad_mcts_state(str(mcts_state))
        return [i for i, c in enumerate(padded) if c != str(self.empty)]

    def _encode_list_state(self, dest_np: np.ndarray, state: Matrix, who_move: int) -> None:
        """
        In-place encodes list state into the zero numpy array
        :param dest_np: dest array, expected to be zero
        :param state_list: state of the game in the list form
        :param who_move: player index (game.PLAYER_WHITE or game.PLAYER_BLACK) who to move
        """
        assert dest_np.shape == self.obs_shape

        for row_idx, row in enumerate(state):
            for col_idx, cell in enumerate(row):
                if cell == who_move:
                    dest_np[0, row_idx, col_idx] = 1.0
                else:
                    dest_np[1, row_idx, col_idx] = 1.0

    def states_to_training_batch(self, state_ints: List[int], who_moves_lists: List[int]) -> List[List]:
        """[summary]

        Args:
            state_ints (List[int]): [description]
            who_moves_lists (List[int]): [description]

        Returns:
            List[List]: [description]
        """
        batch_size = len(state_ints)
        batch = np.zeros((batch_size,) + self.obs_shape, dtype=np.float32)
        for idx, (state, who_move) in enumerate(zip(state_ints, who_moves_lists)):
            converted_state = self.convert_mcts_state_to_list_state(state)
            self._encode_list_state(batch[idx], converted_state, who_move)
        return batch

    def move(self, mcts_state: int, move: int, player: int) -> Tuple[int, bool]:
        """[summary]

        Args:
            mcts_state (int): [description]
            move (int): [description]
            player (int): [description]

        Returns:
            Tuple[int, bool]: [description]
        """
        assert player == self.player_white or player == self.player_black
        assert move >= 0 and move <= self.action_space

        board = self.convert_mcts_state_to_list_state(mcts_state)
        row_idx, col_idx = divmod(move, self.board_len)
        board[row_idx][col_idx] = player
        won = tictactoe_helpers.check_win(
            board, (row_idx, col_idx), self.k_to_win, player)
        new_mcts_state = self.encode_game_state(board)
        return won, new_mcts_state

    def render(self, mcts_state: int) -> str:
        """[summary]

        Args:
            mcts_state (int): [description]

        Returns:
            str: [description]
        """
        pass
