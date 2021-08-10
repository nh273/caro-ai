import pytest
from unittest.mock import MagicMock, patch
from lib.mcts import MCTS


@pytest.fixture
def tree():
    """Let's mock a game with 2 possible actions: 0 & 1
    Let's construct a tree that started from state 1,
    took action 1 once and that led to state 2,
    then from state 2 take action 0 that led to state 3
    """
    mock_game = MagicMock()
    tree = MCTS(mock_game)
    tree.visit_count = {1: [0, 1], 2: [1, 0], 3: [0, 0]}
    tree.value = {1: [0.0, 0.5], 2: [0.6, 0.0], 3: [0.0, 0.0]}
    tree.value_avg = {1: [0.0, 0.5], 2: [0.6, 0.0], 3: [0.0, 0.0]}
    # Remember prior probabilities of actions at each state sum to 1
    # State 3 has not been visited so everything is 0 except prior probs
    # queried from neural network
    tree.probs = {1: [0.1, 0.9], 2: [0.8, 0.2], 3: [0.7, 0.3]}
    return tree


class TestBackup:
    def test_back_up(self, tree):
        value = 0.2
        states = [1, 2, 3]
        # Let's say we take the same actions again (1 --1--> 2 --0--> 3)
        # then 0
        actions = [1, 0, 0]
        tree._backup(value, states, actions)
        assert tree.visit_count == {1: [0, 2], 2: [2, 0], 3: [1, 0]}
        # Remember to flip the sign of value at each turn
        assert tree.value == {1: [0.0, 0.3], 2: [0.8, 0.0], 3: [-0.2, 0.0]}
        # Mean value over visit_count
        assert tree.value_avg == {
            1: [0.0, 0.15], 2: [0.4, 0.0], 3: [-0.2, 0.0]}
