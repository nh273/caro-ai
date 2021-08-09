from lib.game.connect_four.connect_four import ConnectFour
from lib.game.tictactoe.tictactoe import TicTacToe


def add_game_argument(parser):
    """Adding a --game argument to the argument parser with the available games

    Args:
        parser (ArgumentParser): The initialized parser
    """
    parser.add_argument("-g", "--game", required=True, choices=['0', '1'],
                        help="The type of game. 0: Connect4, 1: TicTacToe")


def get_game(args):
    """Return the game instance specified in args

    Args:
        args: parsed args from ArgumentParser)
    """
    game_type = args.game
    return ConnectFour() if game_type == '0' else TicTacToe()
