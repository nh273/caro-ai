#!/usr/bin/env python3
import os
import sys
import time
import ptan
import random
import argparse
import collections
from typing import Union

from lib.model import Net
from lib.mcts import MCTS
from lib.utils import play_game
from lib.game.game import BaseGame
from lib.game.connect_four.connect_four import ConnectFour
from lib.game.tictactoe.tictactoe import TicTacToe


from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import torch.nn.functional as F


PLAY_EPISODES = 1  # 25
MCTS_SEARCHES = 10
MCTS_BATCH_SIZE = 8
REPLAY_BUFFER = 5000  # 30000
LEARNING_RATE = 0.1
BATCH_SIZE = 256
TRAIN_ROUNDS = 10
MIN_REPLAY_TO_TRAIN = 2000  # 10000

BEST_NET_WIN_RATIO = 0.60

EVALUATE_EVERY_STEP = 100
EVALUATION_ROUNDS = 20
STEPS_BEFORE_TAU_0 = 10

Optimizer = torch.optim.Optimizer


def self_play(game: BaseGame, mcts_store: MCTS, replay_buffer: Union[collections.deque, None],
              model: Net, tb_tracker, device: str) -> None:
    """Let the (current best) model play against itself to generate training data.
    Store the moves into a replay buffer.

    Args:
        game (Game): The game that the net is being trained to play
        mcts_store (MCTS): Monte Carlo Tree of fully-played games' states & outcomes
        replay_buffer (deque): Queue of moves and predicted values made by the
            current best net
        model (Net): Neural net trained to play the game
        tb_tracker (Tensorflow Board tracker) to collect stats
        device (str): cpu or gpu for PyTorch
    """
    t = time.time()
    prev_nodes = len(mcts_store)
    game_steps = 0
    for _ in range(PLAY_EPISODES):
        _, steps = play_game(game, mcts_store, replay_buffer,
                             best_net.target_model, best_net.target_model,
                             steps_before_tau_0=STEPS_BEFORE_TAU_0, mcts_searches=MCTS_SEARCHES,
                             mcts_batch_size=MCTS_BATCH_SIZE, device=device)
        game_steps += steps
    game_nodes = len(mcts_store) - prev_nodes
    dt = time.time() - t
    speed_steps = game_steps / dt
    speed_nodes = game_nodes / dt
    tb_tracker.track("speed_steps", speed_steps, step_idx)
    tb_tracker.track("speed_nodes", speed_nodes, step_idx)
    sys.stdout.flush()
    buffer_len = len(replay_buffer) if replay_buffer else 0
    print("Step %d, steps %3d, leaves %4d, steps/s %5.2f, leaves/s %6.2f, best_idx %d, replay %d" % (
        step_idx, game_steps, game_nodes, speed_steps, speed_nodes, best_idx, buffer_len),
        end='\r')


def train_neural_net(game: BaseGame, replay_buffer: collections.deque, optimizer: Optimizer,
                     tb_tracker, device: str) -> None:
    """Give a replay buffer that is sufficiently large, train the neural net
    using data from replay buffer in batches.

    Args:
        game (Game): The type of game that the model is being trained on
        replay_buffer (deque): Queue of steps collected during self-play. This
            is data used to train the neural net
        optimizer (PyTorch optimizer): optimizer to perform gradient descent
            & update neural net tensor values
        tb_tracker (TensorflowBoard): tracker that write model stats to Tensorflow
            Board
        device (str): cpu or gpu (for PyTorch)
    """
    sum_loss = 0.0
    sum_value_loss = 0.0
    sum_policy_loss = 0.0

    for _ in range(TRAIN_ROUNDS):
        # PyTorch is trained in batches. We process data batches here into tensors
        # that can be fed into the neural net for training
        batch = random.sample(replay_buffer, BATCH_SIZE)
        batch_states, batch_who_moves, batch_probs, batch_values = zip(
            *batch)
        states_v = game.states_to_training_batch(
            batch_states, batch_who_moves)
        states_tensor = torch.tensor(states_v).to(device)

        optimizer.zero_grad()
        probs_v = torch.FloatTensor(batch_probs).to(device)
        values_v = torch.FloatTensor(batch_values).to(device)
        out_logits_v, out_values_v = net(states_tensor)

        # calculate MSE between model's value prediction vs actual result
        loss_value_v = F.mse_loss(out_values_v.squeeze(-1), values_v)

        # calculate cross-entropy between model's policy probs and probs
        # sampled from MCTS
        loss_policy_v = -F.log_softmax(out_logits_v, dim=1) * probs_v
        loss_policy_v = loss_policy_v.sum(dim=1).mean()

        loss_v = loss_policy_v + loss_value_v
        # back propagation & gradient descent
        loss_v.backward()
        optimizer.step()
        sum_loss += loss_v.item()
        sum_value_loss += loss_value_v.item()
        sum_policy_loss += loss_policy_v.item()

    tb_tracker.track("loss_total", sum_loss / TRAIN_ROUNDS, step_idx)
    tb_tracker.track("loss_value", sum_value_loss /
                     TRAIN_ROUNDS, step_idx)
    tb_tracker.track("loss_policy", sum_policy_loss /
                     TRAIN_ROUNDS, step_idx)


def evaluate(game: BaseGame, challenger: Net, champion: Net,
             rounds: int, device: str = "cpu") -> float:
    """Evaluate performance of 2 neural nets trained to play game by letting them
    play rounds against each other.

    Args:
        game (Game): The game that the net is being trained to play
        challenger, champion (Net): 2 instances of PyTorch neural net
            trained to play game to compare performance
        rounds (int): Number of rounds that the nets will play
        device (str, optional): [description]. Defaults to "cpu".

    Returns:
        [float]: Proportions of win by challenger
    """
    challenger_win, champion_win, draw = 0, 0, 0
    mcts_stores = [MCTS(game), MCTS(game)]

    for r_idx in range(rounds):
        r, _ = play_game(game=game, mcts_stores=mcts_stores, replay_buffer=None,
                         net1=challenger, net2=champion,
                         steps_before_tau_0=0, mcts_searches=20, mcts_batch_size=16,
                         device=device)
        if r < -0.5:
            champion_win += 1
        elif r > 0.5:
            challenger_win += 1
        elif r == 0:
            draw += 1
    return challenger_win / (challenger_win + champion_win + draw)


def parse_args():
    """Add and parse arguments.
    Returns:
        [args]: args object with parsed arguments in its fields
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable CUDA")
    parser.add_argument("-g", "--game", required=True, choices=['0', '1'],
                        help="The type of game being trained. 0: Connect4, 1: TicTacToe")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if args.cuda else "cpu"

    saves_path = os.path.join("saves", args.name)
    os.makedirs(saves_path, exist_ok=True)
    writer = SummaryWriter(comment="-" + args.name)

    game_type = args.game
    game = ConnectFour() if game_type == '0' else TicTacToe()
    model_shape = game.obs_shape

    net = Net(input_shape=model_shape,
              actions_n=game.action_space).to(device)
    best_net = ptan.agent.TargetNet(net)
    print(net)

    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)

    replay_buffer = collections.deque(maxlen=REPLAY_BUFFER)
    mcts_store = MCTS(game)
    step_idx = 0
    best_idx = 0

    with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
        # Theoretically training loop can continue forever
        # to produce better & better agents
        while True:
            self_play(game, mcts_store, replay_buffer,
                      best_net.target_model, tb_tracker, device)
            step_idx += 1

            if len(replay_buffer) < MIN_REPLAY_TO_TRAIN:
                continue

            # Replay buffer has sufficient data. Train the net
            train_neural_net(game, replay_buffer,
                             optimizer, tb_tracker, device)

            # evaluate net, then replace best net if performance is satisfactory
            if step_idx % EVALUATE_EVERY_STEP == 0:
                win_ratio = evaluate(game,
                                     net, best_net.target_model, rounds=EVALUATION_ROUNDS, device=device)
                print("Net evaluated, win ratio = %.2f" % win_ratio)
                writer.add_scalar("eval_win_ratio", win_ratio, step_idx)
                if win_ratio > BEST_NET_WIN_RATIO:
                    print("Net is better than cur best, sync")
                    best_net.sync()
                    best_idx += 1
                    file_name = os.path.join(
                        saves_path, "best_%03d_%05d.dat" % (best_idx, step_idx))
                    torch.save(net.state_dict(), file_name)
                    mcts_store.clear()
