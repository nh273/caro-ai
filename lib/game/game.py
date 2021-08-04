"""Defining the interface for Game class
"""

from abc import ABC, abstractmethod
from typing import Tuple, List


class BaseGame(ABC):
    @property
    @abstractmethod
    def initial_state(self) -> int:
        """The initial state of the game in MCTS form. This is used in
        utils.play_game to start the MCTS loop.
        """
        pass

    @property
    @abstractmethod
    def obs_shape(self) -> Tuple[int, ...]:
        """The shape of the neural network form of the game state.
        This should be a tuple of ints. e.g. For a game of Tic-Tac-Toe it's
        likely (2, 3, 3) i.e. the game state being fed into the neural net
        is a 2x3x3 tensor: 2 player x 3x3 board (board viewed from each player's
        side)
        """
        pass

    @property
    @abstractmethod
    def action_space(self) -> int:
        """The total count of all possible actions that can be performed.
        This is a constant for each game that represents all actions,
        regardless of whether they are valid at each game state.

        Used to initialize MCTS nodes (determine how large the values, avg_values &
        visit count vectors are)

        Returns:
            (int): count of total possible actions
        """
        pass

    @abstractmethod
    def possible_moves(self, mcts_state: int) -> List:
        """[summary]

        Args:
            mcts_state (Hashable): [description]

        Returns:
            Iterable: [description]
        """
        pass

    @abstractmethod
    def invalid_moves(self, mcts_state: int) -> List:
        """[summary]

        Args:
            mcts_state (Hashable): [description]

        Returns:
            Iterable: [description]
        """
        pass

    @abstractmethod
    def states_to_training_batch(self, state_lists: List[List], who_moves_lists: List[int]) -> List[List]:
        """[summary]

        Args:
            state_lists (List[List]): [description]
            who_moves_lists (List[int]): [description]

        Returns:
            List[List]: [description]
        """
        pass

    @abstractmethod
    def move(self, mcts_state: int, move: int, player: int) -> Tuple[int, bool]:
        """[summary]

        Args:
            mcts_state (Hashable): [description]
            move (int): [description]
            player (int): [description]

        Returns:
            Tuple[Hashable, bool]: [description]
        """
        pass

    @abstractmethod
    def render(self, mcts_state: int) -> str:
        """[summary]

        Args:
            mcts_state (Hashable): [description]

        Returns:
            str: [description]
        """
        pass
