import collections
import numpy as np
from typing import Union, Tuple, Dict
import torch
from lib import mcts, model
from lib.game.game import BaseGame


def update_counts(counts_dict: Dict, key: Union[str, Tuple[str, str]], counts: Tuple[int, int, int]) -> None:
    """Update counts_dict with win, lose, draw from counts if key exist.
    Else initialize new entry with 0, 0, 0
    Key can be a string representing a model name, or a tuple representing
    2 dueling models

    Args:
        counts_dict (Dict): Dictionary that keep track of wins, losses, and draws
        key (Union[str, Tuple[str, str]])
        counts (Tuple[int, int, int]): Win, Losses, and Draws
    """
    v = counts_dict.get(key, (0, 0, 0))
    res = (v[0] + counts[0], v[1] + counts[1], v[2] + counts[2])
    counts_dict[key] = res


def play_game(game: BaseGame, mcts_stores, replay_buffer: Union[collections.deque, None],
              net1: model.Net, net2: model.Net,
              steps_before_tau_0: int, mcts_searches: int, mcts_batch_size: int,
              net1_plays_first: bool = None, device: str = "cpu"):
    """
    Play one single game, memorizing transitions into the replay buffer
    :param net1: player1
    :param net2: player2

    Args:
        game ([type]): [description]
        mcts_stores ([type]): could be None or single MCTS or two MCTSes for individual net
        replay_buffer (deque): queue with (state, probs, values), if None, nothing is stored
        net1 (model.Net): [description]
        net2 (model.Net): [description]
        steps_before_tau_0 ([type]): [description]
        mcts_searches (int): [description]
        mcts_batch_size (int): [description]
        net1_plays_first (bool, optional): [description]. Defaults to None.
        device (str, optional): [description]. Defaults to "cpu".

    Returns:
        [int]: value for the game in respect to net_1 (+1 if p1 won, -1 if lost, 0 if draw)
        [int]:
    """
    assert isinstance(replay_buffer, (collections.deque, type(None)))
    assert isinstance(mcts_stores, (mcts.MCTS, type(None), list))
    assert isinstance(net1, model.Net)
    assert isinstance(net2, model.Net)
    assert isinstance(steps_before_tau_0, int) and steps_before_tau_0 >= 0
    assert isinstance(mcts_searches, int) and mcts_searches > 0
    assert isinstance(mcts_batch_size, int) and mcts_batch_size > 0

    if mcts_stores is None:
        mcts_stores = [mcts.MCTS(game), mcts.MCTS(game)]
    elif isinstance(mcts_stores, mcts.MCTS):
        mcts_stores = [mcts_stores, mcts_stores]

    state = game.initial_state
    nets = [net1, net2]
    if net1_plays_first is None:
        cur_player = np.random.choice(2)
    else:
        cur_player = 0 if net1_plays_first else 1
    step = 0
    tau = 1 if steps_before_tau_0 > 0 else 0
    game_history = []

    result = None
    net1_result = None

    while result is None:
        mcts_stores[cur_player].search_batch(
            mcts_searches, mcts_batch_size, state,
            cur_player, nets[cur_player], device=device)
        probs, _ = mcts_stores[cur_player].get_policy_value(
            state, tau=tau)
        game_history.append((state, cur_player, probs))
        action = np.random.choice(game.action_space, p=probs)
        if action not in game.possible_moves(state):
            print("Impossible action selected")
        state, won = game.move(state, action, cur_player)
        if won:
            result = 1
            net1_result = 1 if cur_player == 0 else -1
            break
        cur_player = 1-cur_player
        # check the draw case
        if len(game.possible_moves(state)) == 0:
            result = 0
            net1_result = 0
            break
        step += 1
        if step >= steps_before_tau_0:
            tau = 0

    if replay_buffer is not None:
        for state, cur_player, probs in reversed(game_history):
            replay_buffer.append(
                (state, cur_player, probs, result)
            )
            result = -result

    return net1_result, step


class TBMeanTracker:
    """
    TensorBoard value tracker: allows to batch fixed amount of historical values and write their mean into TB
    Designed and tested with pytorch-tensorboard in mind
    """

    def __init__(self, writer, batch_size):
        """
        :param writer: writer with close() and add_scalar() methods
        :param batch_size: integer size of batch to track
        """
        assert isinstance(batch_size, int)
        assert writer is not None
        self.writer = writer
        self.batch_size = batch_size

    def __enter__(self):
        self._batches = collections.defaultdict(list)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()

    @staticmethod
    def _as_float(value):
        assert isinstance(value, (float, int, np.ndarray, np.generic,
                          torch.autograd.Variable)) or torch.is_tensor(value)
        tensor_val = None
        if isinstance(value, torch.autograd.Variable):
            tensor_val = value.data
        elif torch.is_tensor(value):
            tensor_val = value

        if tensor_val is not None:
            return tensor_val.float().mean().item()
        elif isinstance(value, np.ndarray):
            return float(np.mean(value))
        else:
            return float(value)

    def track(self, param_name, value, iter_index):
        assert isinstance(param_name, str)
        assert isinstance(iter_index, int)

        data = self._batches[param_name]
        data.append(self._as_float(value))

        if len(data) >= self.batch_size:
            self.writer.add_scalar(param_name, np.mean(data), iter_index)
            data.clear()
