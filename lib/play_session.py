import torch
import numpy as np
import config as cfg
from lib import model, mcts


class Session:
    def __init__(self, game, model_file, player_moves_first):
        self.game = game
        self.BOT_PLAYER = game.player_black
        self.USER_PLAYER = game.player_white
        self.model_file = model_file
        self.model = model.Net(
            input_shape=game.obs_shape, actions_n=game.action_space)
        self.model.load_state_dict(torch.load(
            model_file, map_location=lambda storage, loc: storage))
        self.state = game.initial_state
        self.value = None
        self.player_moves_first = player_moves_first
        self.moves = []
        self.mcts_store = mcts.MCTS(game)

    def move_player(self, move: int) -> bool:
        self.moves.append(move)
        self.state, won = self.game.move(self.state, move, self.USER_PLAYER)
        return won

    def move_bot(self) -> bool:
        self.mcts_store.search_batch(
            cfg.BOT_MCTS_SEARCHES, cfg.BOT_MCTS_BATCH_SIZE, self.state, self.BOT_PLAYER, self.model)
        probs, values = self.mcts_store.get_policy_value(self.state, tau=0)
        action = np.random.choice(self.game.action_space, p=probs)
        self.value = values[action]
        self.moves.append(action)
        self.state, won = self.game.move(self.state, action, self.BOT_PLAYER)
        return won

    def is_valid_move(self, move: int) -> bool:
        return move in self.game.possible_moves(self.state)

    def is_draw(self) -> bool:
        return len(self.game.possible_moves(self.state)) == 0

    def render(self) -> str:
        board = self.game.render(self.state)
        extra = ""
        if self.value is not None:
            extra = "Position evaluation: %.2f\n" % float(self.value)
        return extra + "<pre>%s</pre>" % board
