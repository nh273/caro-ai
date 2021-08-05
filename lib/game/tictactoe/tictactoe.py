from lib.game.game import BaseGame


class NMKGame(BaseGame):
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
        self.player_X = 0
        self.player_0 = 1

    @staticmethod
    def flatten_nested_list(nested_list: List[List]) -> List:
        """[summary]

        Args:
            nested_list (List[List]): [description]

        Returns:
            List: [description]
        """
        return [item for sublist in nested_list for item in sublist]

    def encode_game_state(self, state_list: List[List[int]]) -> int:
        """[summary]

        Args:
            state_list (List[List[int]]): [description]

        Returns:
            int: [description]
        """
        flattened = self.flatten_nested_list(state_list)
        return

    @property
    def initial_state(self) -> int:
        """The initial state of the game in MCTS form. This is used in
        utils.play_game to start the MCTS loop.
        """
        pass

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        """The shape of the neural network form of the game state.
        This should be a tuple of ints. e.g. For a game of Tic-Tac-Toe it's
        likely (2, 3, 3) i.e. the game state being fed into the neural net
        is a 2x3x3 tensor: 2 player x 3x3 board (board viewed from each player's
        side)
        """
        pass

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

    def convert_mcts_state_to_nn_state(self, mcts_state: int) -> List:
        """[summary]

        Args:
            mcts_state (Hashable): [description]

        Returns:
            (List): [description]
        """
        pass

    def possible_moves(self, mcts_state: int) -> List:
        """[summary]

        Args:
            mcts_state (Hashable): [description]

        Returns:
            Iterable: [description]
        """
        pass

    def invalid_moves(self, mcts_state: int) -> List:
        """[summary]

        Args:
            mcts_state (Hashable): [description]

        Returns:
            Iterable: [description]
        """
        pass

    def states_to_training_batch(self, state_ints: List[int], who_moves_lists: List[int]) -> List[List]:
        """[summary]

        Args:
            state_ints (List[int]): [description]
            who_moves_lists (List[int]): [description]

        Returns:
            List[List]: [description]
        """
        pass

    def move(self, mcts_state: int, move: int, player: int) -> Tuple[int, bool]:
        """[summary]

        Args:
            mcts_state (int): [description]
            move (int): [description]
            player (int): [description]

        Returns:
            Tuple[int, bool]: [description]
        """
        pass

    def render(self, mcts_state: int) -> str:
        """[summary]

        Args:
            mcts_state (int): [description]

        Returns:
            str: [description]
        """
        pass
