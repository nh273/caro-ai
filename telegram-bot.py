#!/usr/bin/env python3
# This module requires python-telegram-bot
import torch
import os
import sys
import glob
import json
import time
import datetime
import random
import logging
import numpy as np
import configparser
import argparse

from lib import model, mcts, utils
from lib.game.connect_four.connect_four import ConnectFour
from lib.game.tictactoe.tictactoe import TicTacToe

MCTS_SEARCHES = 100
MCTS_BATCH_SIZE = 5

try:
    import telegram.ext
    from telegram.error import TimedOut
except ImportError:
    print("You need python-telegram-bot package installed to start the bot")
    sys.exit()


# Configuration file with the following contents
# [telegram]
# api=API_KEY
CONFIG_DEFAULT = "./.config/bot.ini"

log = logging.getLogger("telegram")


class Session:
    def __init__(self, game, model_file, player_moves_first, player_id):
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
        self.player_id = player_id
        self.moves = []
        self.mcts_store = mcts.MCTS(game)

    def move_player(self, move: int) -> bool:
        self.moves.append(move)
        self.state, won = self.game.move(self.state, move, self.USER_PLAYER)
        return won

    def move_bot(self) -> bool:
        self.mcts_store.search_batch(
            MCTS_SEARCHES, MCTS_BATCH_SIZE, self.state, self.BOT_PLAYER, self.model)
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


class PlayerBot:
    def __init__(self, game, models_dir, log_file):
        self.game = game
        self.sessions = {}
        self.models_dir = models_dir
        self.models = self._read_models(models_dir)
        self.log_file = log_file
        self.leaderboard = {}
        self._read_leaderboard(log_file)

    def _read_models(self, models_dir):
        result = {}
        for idx, name in enumerate(sorted(glob.glob(os.path.join(models_dir, "*.dat")))):
            result[idx] = name
        return result

    def _read_leaderboard(self, log_file):
        if not os.path.exists(log_file):
            return
        with open(log_file, 'rt', encoding='utf-8') as fd:
            for l in fd:
                data = json.loads(l)
                bot_name = os.path.basename(data['model_file'])
                user_name = data['player_id'].split(':')[0]
                bot_score = data['bot_score']
                self._update_leaderboard(bot_score, bot_name, user_name)

    def _update_leaderboard(self, bot_score, bot_name, user_name):
        if bot_score > 0.5:
            utils.update_counts(self.leaderboard, bot_name, (1, 0, 0))
            utils.update_counts(self.leaderboard, user_name, (0, 1, 0))
        elif bot_score < -0.5:
            utils.update_counts(self.leaderboard, bot_name, (0, 1, 0))
            utils.update_counts(self.leaderboard, user_name, (1, 0, 0))
        else:
            utils.update_counts(self.leaderboard, bot_name, (0, 0, 1))
            utils.update_counts(self.leaderboard, user_name, (0, 0, 1))

    def _save_log(self, session, bot_score):
        self._update_leaderboard(bot_score, os.path.basename(session.model_file),
                                 session.player_id.split(':')[0])
        data = {
            "ts": time.time(),
            "time": datetime.datetime.utcnow().isoformat(),
            "bot_score": bot_score,
            "model_file": session.model_file,
            "player_id": session.player_id,
            "player_moves_first": session.player_moves_first,
            "moves": session.moves,
            "state": session.state
        }
        with open(self.log_file, "a+t", encoding='utf-8') as f:
            f.write(json.dumps(data, sort_keys=True) + '\n')

    def command_help(self, bot, update):
        bot.send_message(chat_id=update.message.chat_id, parse_mode="HTML", disable_web_page_preview=True,
                         text="""
This a <a href="https://en.wikipedia.org/wiki/Connect_Four">4-in-a-row</a> game bot trained with AlphaGo Zero method for the <a href="https://www.packtpub.com/big-data-and-business-intelligence/practical-deep-reinforcement-learning">Practical Deep Reinforcement Learning</a> book.

<b>Welcome!</b>

This bot understands the following commands:
<b>/list</b> to list available pre-trained models (the higher the ID, the stronger the play)
<b>/play MODEL_ID</b> to start the new game against the specified model
<b>/top</b> show leaderboard

During the game, your moves are numbers of columns to drop the disk.
""")

    def command_list(self, bot, update):
        if len(self.models) == 0:
            reply = ["There are no models currently available, sorry!"]
        else:
            reply = ["The list of available models with their IDs"]
            for idx, name in sorted(self.models.items()):
                reply.append("<b>%d</b>: %s" % (idx, os.path.basename(name)))

        bot.send_message(chat_id=update.message.chat_id,
                         text="\n".join(reply), parse_mode="HTML")

    def command_play(self, bot, update, args):
        chat_id = update.message.chat_id
        player_id = "%s:%s" % (
            update.message.from_user.username, update.message.from_user.id)
        try:
            model_id = int(args[0])
        except ValueError:
            bot.send_message(
                chat_id=chat_id, text="Wrong argumants! Use '/play <MODEL_ID>, to start the game")
            return

        if model_id not in self.models:
            bot.send_message(
                chat_id=chat_id, text="There is no such model, use /list command to get list of IDs")
            return

        if chat_id in self.sessions:
            bot.send_message(
                chat_id=chat_id, text="You already have the game in progress, it will be discarded")
            del self.sessions[chat_id]

        player_moves = random.choice([False, True])
        session = Session(
            self.game, self.models[model_id], player_moves, player_id)
        self.sessions[chat_id] = session
        if player_moves:
            bot.send_message(
                chat_id=chat_id, text=f"Your move is first (you're playing with O), please give the position of your move - single number from 0 to {self.game.action_space}")
        else:
            bot.send_message(
                chat_id=chat_id, text="The first move is mine (I'm playing with X), moving...")
            session.move_bot()
        bot.send_message(
            chat_id=chat_id, text=session.render(), parse_mode="HTML")

    def text(self, bot, update):
        chat_id = update.message.chat_id

        if chat_id not in self.sessions:
            bot.send_message(chat_id=chat_id, text="You have no game in progress. Start it with <b>/play MODEL_ID</b> "
                                                   "(or use <b>/help</b> to see the list of commands)",
                             parse_mode='HTML')
            return
        session = self.sessions[chat_id]

        try:
            move = int(update.message.text)
        except ValueError:
            bot.send_message(chat_id=chat_id, text="I don't understand. In play mode you can give a number "
                                                   "to specify your move.")
            return

        if not session.is_valid_move(move):
            bot.send_message(
                chat_id=chat_id, text="Move %d is invalid!" % move)
            return

        won = session.move_player(move)
        if won:
            bot.send_message(chat_id=chat_id, text="You won! Congratulations!")
            self._save_log(session, bot_score=-1)
            del self.sessions[chat_id]
            return

        won = session.move_bot()
        bot.send_message(
            chat_id=chat_id, text=session.render(), parse_mode="HTML")

        if won:
            bot.send_message(chat_id=chat_id, text="I won! Wheeee!")
            self._save_log(session, bot_score=1)
            del self.sessions[chat_id]
        # checking for a draw
        if session.is_draw():
            bot.send_message(
                chat_id=chat_id, text="Draw position. 1:1, see ya!")
            self._save_log(session, bot_score=0)
            del self.sessions[chat_id]

    def error(self, bot, update, error):
        try:
            raise error
        except TimedOut:
            log.info("Timed out error")

    def command_top(self, bot, update):
        res = ["Leader board"]
        items = list(self.leaderboard.items())
        items.sort(reverse=True, key=lambda p: p[1][0])
        for user, (wins, losses, draws) in items:
            res.append("%20s: won=%d, lost=%d, draw=%d" %
                       (user[:20], wins, losses, draws))
        l = "\n".join(res)
        bot.send_message(chat_id=update.message.chat_id,
                         text="<pre>" + l + "</pre>", parse_mode="HTML")

    def command_refresh(self, bot, update):
        self.models = self._read_models(self.models_dir)
        bot.send_message(chat_id=update.message.chat_id,
                         text="Models reloaded, %d files have found" % len(self.models))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_DEFAULT,
                        help="Configuration file for the bot, default=" + CONFIG_DEFAULT)
    parser.add_argument("-m", "--models", required=True,
                        help="Directory name with models to serve")
    parser.add_argument("-l", "--log", default=f'logs/{time.strftime("%Y%m%d-%H%M%S")}.log',
                        help="Log name to keep the games and leaderboard")
    parser.add_argument("-g", "--game", required=True, choices=['0', '1'],
                        help="The type of game being trained. 0: Connect4, 1: TicTacToe")
    args = parser.parse_args()

    conf = configparser.ConfigParser()
    if not conf.read(os.path.expanduser(args.config)):
        log.error("Configuration file %s not found", args.config)
        sys.exit()

    game = ConnectFour() if args.game == '0' else TicTacToe()
    player_bot = PlayerBot(game, args.models, args.log)

    updater = telegram.ext.Updater(conf['telegram']['api'])
    updater.dispatcher.add_handler(
        telegram.ext.CommandHandler('help', player_bot.command_help))
    updater.dispatcher.add_handler(
        telegram.ext.CommandHandler('list', player_bot.command_list))
    updater.dispatcher.add_handler(
        telegram.ext.CommandHandler('top', player_bot.command_top))
    updater.dispatcher.add_handler(telegram.ext.CommandHandler(
        'play', player_bot.command_play, pass_args=True))
    updater.dispatcher.add_handler(telegram.ext.CommandHandler(
        'refresh', player_bot.command_refresh))
    updater.dispatcher.add_handler(telegram.ext.MessageHandler(
        telegram.ext.Filters.text, player_bot.text))
    updater.dispatcher.add_error_handler(player_bot.error)

    log.info("Bot initialized, started serving")
    updater.start_polling()
    updater.idle()
