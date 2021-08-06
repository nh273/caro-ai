"""
Monte-Carlo Tree Search
"""
import math as m
import numpy as np

from lib.model import Net
from lib.game.game import BaseGame

import torch
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional

StateInt = int
VisitCount = Dict[StateInt, List[int]]
Value = Dict[StateInt, List[float]]
ValueAverage = Dict[StateInt, List[float]]
Probs = Dict[StateInt, List[float]]


class MCTS:
    """
    Class keeps statistics for every state encountered during the search

    """

    def __init__(self, game: BaseGame, c_puct: float = 1.0):
        self.c_puct = c_puct
        # count of visits, state_int -> [N(s, a)]
        self.visit_count: VisitCount = {}
        # total value of the state's act, state_int -> [W(s, a)]
        self.value: Value = {}
        # average value of actions, state_int -> [Q(s, a)]
        self.value_avg: ValueAverage = {}
        # prior probability of actions, state_int -> [P(s,a)]
        self.probs: Probs = {}
        self.game = game

    def clear(self):
        self.visit_count.clear()
        self.value.clear()
        self.value_avg.clear()
        self.probs.clear()

    def __len__(self):
        return len(self.value)

    def _add_noise(self, probs: List[float]) -> List[float]:
        """Add Dirichlet noise to action probabilities

        Args:
            probs (list): List of probabilities of actions
        """
        # The alpha value for dirichlet distribution (0.03)
        # and exploration coefficient (0.25) are
        # taken from the AlphaZero paper for Go.
        alpha = 0.03
        explore = 0.25
        noises = np.random.dirichlet(
            [alpha] * self.game.action_space)
        probs_with_noise = [
            (1 - explore) * prob + explore * noise
            for prob, noise in zip(probs, noises)
        ]
        return probs_with_noise

    def _calculate_upper_bound(self, values_avg: List[float], probs: List[float],
                               counts: List[int]) -> List[float]:
        """Calculate a score for each action at the current the game state
        from average values, probabilities & counts.


        Args:
            values_avg (List[float]): Q(s, a) in the paper — The mean action value.
                This is the average game result across current simulations that took action a.
            probs (List[float]): P(s,a) — The prior probabilities as fetched from the network.
            counts (List[int]): N(s,a) — The visit count, or number of times we’ve taken
                this action given this sate during current simulations
        Return:
            (List[float])
        """
        total_sqrt = m.sqrt(sum(counts))
        return [
            value + self.c_puct * prob * total_sqrt/(1+count)
            for value, prob, count in
            zip(values_avg, probs, counts)
        ]

    def _mask_invalid_actions(self, scores: List[float], cur_state: int) -> None:
        """Inplace mask invalid actions from the neural net by penalizing it with
        negative infinity scores.

        Args:
            scores (list of float): List of unmasked scores for each action
            cur_state (int): State of the game
        """
        invalid_actions = self.game.invalid_moves(cur_state)
        for invalid in invalid_actions:
            scores[invalid] = -np.inf

    def find_leaf(self, state_int: StateInt,
                  player: int) -> Tuple[Optional[float], StateInt, int, List[StateInt], List[int]]:
        """
        Traverse the tree until the end of game or leaf node
        (state which we have not seen before)
        :param state_int: root node state
        :param player: player to move
        :return: tuple of (value, leaf_state, player, states, actions)
        1. value: None if leaf node, otherwise equals to the game outcome for the player at leaf
        2. leaf_state: state_int of the last state
        3. player: player at the leaf node
        4. states: list of states traversed
        5. list of actions taken

        Args:
            state_int (int): [description]
            player (int): [description]

        Returns:
            Tuple[float, int, int, List, List[int]]: [description]
        """
        states = []
        actions = []
        cur_state = state_int
        cur_player = player
        value = None

        while not self.is_leaf(cur_state):
            states.append(cur_state)

            counts = self.visit_count[cur_state]
            probs = self.probs[cur_state]
            values_avg = self.value_avg[cur_state]

            # In the root node (first move) add the Dirichlet noise to the probs
            # to encourage exploration. Not needed for subsequent moves as will
            # be sampling from a distribution of actions
            if cur_state == state_int:
                probs = self._add_noise(probs)
            scores = self._calculate_upper_bound(values_avg, probs, counts)
            self._mask_invalid_actions(scores, cur_state)
            action = int(np.argmax(scores))
            actions.append(action)
            cur_state, won = self.game.move(
                cur_state, action, cur_player)
            if won:
                # if somebody won the game, the value of the final state is -1 (as it is on opponent's turn)
                value = -1.0
            cur_player = 1-cur_player
            # check for draw
            if value is None and len(self.game.possible_moves(cur_state)) == 0:
                value = 0.0

        return value, cur_state, cur_player, states, actions

    def is_leaf(self, state_int: StateInt) -> bool:
        """[summary]

        Args:
            state_int (int): [description]

        Returns:
            [type]: [description]
        """
        return state_int not in self.probs

    def search_batch(self, count: int, batch_size: int, state_int: StateInt,
                     player: int, net: Net, device: str = "cpu"):
        """[summary]

        Args:
            count (int): [description]
            batch_size (int): [description]
            state_int (int): [description]
            player (int): [description]
            net (Net): [description]
            device (str, optional): [description]. Defaults to "cpu".
        """
        for _ in range(count):
            self.search_minibatch(batch_size, state_int,
                                  player, net, device)

    def _create_node(self, leaf_state: int, prob: List[float]):
        """[summary]

        Args:
            leaf_state ([type]): [description]
            prob (float):
        """
        action_space = self.game.action_space
        self.visit_count[leaf_state] = [0]*action_space
        self.value[leaf_state] = [0.0]*action_space
        self.value_avg[leaf_state] = [0.0]*action_space
        self.probs[leaf_state] = prob

    def _expand_tree(self, expand_states: List[StateInt], expand_players: List[int],
                     expand_queue: List, backup_queue: List,
                     net: Net, device: str = "cpu") -> None:
        """[summary]

        Args:
            expand_states ([type]): [description]
            expand_players ([type]): [description]
            expand_queue ([type]): [description]
            backup_queue ([type]): [description]
            net():
            device():
        """
        batch_v = self.game.states_to_training_batch(
            expand_states, expand_players)
        batch_tensor = torch.tensor(batch_v).to(device)
        logits_v, values_v = net(batch_tensor)
        probs_v = F.softmax(logits_v, dim=1)
        values = values_v.data.cpu().numpy()[:, 0]
        probs = probs_v.data.cpu().numpy()

        # create the nodes
        for (leaf_state, states, actions), value, prob in zip(expand_queue, values, probs):
            self._create_node(leaf_state, prob)
            backup_queue.append((value, states, actions))

    def _backup(self, value: float, states: List[StateInt], actions: List[int]):
        """Update the existing states with new values, average values, visit counts
        as the MCTS process expands the state tree.

        Args:
            value (float): The value of a game state predicted by the neural net
             or final value of the game (win/lose/draw)
            states (List[int]): List of game states to be backed up
            actions (List[int]): List of actions performed that led to the
             corresponding game states
        """
        # leaf state is not stored in states and actions,
        # so the value of the leaf will be the value of the opponent
        cur_value = -value
        for state_int, action in zip(states[::-1],
                                     actions[::-1]):
            self.visit_count[state_int][action] += 1
            self.value[state_int][action] += cur_value
            # update the average value with new value
            self.value_avg[state_int][action] = (self.value[state_int][action] /
                                                 self.visit_count[state_int][action])
            cur_value = -cur_value  # flip the sign after each turn

    def search_minibatch(self, batch_size: int, state_int: StateInt, player: int,
                         net: Net, device: str = "cpu") -> None:
        """
        Perform several MCTS searches. PyTorch neural net is trained in batches,
        thus it is more convenient to perform the MCTS in batches as well.

        Args:
            batch_size (int): Number of searches in this batch
            state_int (int): Game state in MCTS form
            player (int): The player whose turn it is to make the move
            net (Net): The neural net to get policy value for unencountered states
            device (str, optional): PyTorch device. Defaults to "cpu".
        """
        backup_queue = []
        expand_states = []
        expand_players = []
        expand_queue = []
        planned = set()
        for _ in range(batch_size):
            value, leaf_state, leaf_player, states, actions = \
                self.find_leaf(state_int, player)
            if value is not None:
                backup_queue.append((value, states, actions))
            else:
                if leaf_state not in planned:
                    planned.add(leaf_state)
                    expand_states.append(leaf_state)
                    expand_players.append(leaf_player)
                    expand_queue.append((leaf_state, states,
                                         actions))

        # do expansion of nodes
        if expand_queue:
            self._expand_tree(expand_states, expand_players,
                              expand_queue, backup_queue, net, device)

        # perform backup of the searches
        for value, states, actions in backup_queue:
            self._backup(value, states, actions)

    def get_policy_value(self, state_int: StateInt, tau: int = 1) -> Tuple[List[float], List[float]]:
        """Extract policy and action-values by the state
        :return: (probs, values)

        Args:
            state_int (int): state of the board
            tau (int, optional): A hyper parameter that makes the model
            explores more. Defaults to 1.

        Returns:
            List[float]: List of probabilities for each action in the total
            action space
            List[float]: List of values for each action in the total
            action space
        """
        counts = self.visit_count[state_int]
        if tau == 0:
            probs = [0.0] * self.game.action_space
            probs[np.argmax(counts)] = 1.0
        else:
            counts_adjusted = [count ** (1.0 / tau) for count in counts]
            total = sum(counts_adjusted)
            probs = [count / total for count in counts_adjusted]
        values = self.value_avg[state_int]
        return probs, values
