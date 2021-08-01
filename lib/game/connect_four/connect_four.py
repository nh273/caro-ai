import numpy as np
from lib.game.game import BaseGame


class ConnectFour(BaseGame):
    """
    4-in-a-row game-related functions.

    Field is 6*7 with pieces falling from the top to the bottom. There are two kinds of pieces: black and white,
    which are encoded by 1 (black) and 0 (white).

    There are two representation of the game:
    1. List of 7 lists with elements ordered from the bottom. For example, this field

    0     1
    0     1
    10    1
    10  0 1
    10  1 1
    101 111

    Will be encoded as [
    [1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0],
    [1],
    [],
    [1, 1, 0],
    [1],
    [1, 1, 1, 1, 1, 1]
    ]

    2. integer number consists from:
        a. 7*6 bits (column-wise) encoding the field. Unoccupied bits are zero
        b. 7*3 bits, each 3-bit number encodes amount of free entries on the top.
    In this representation, the field above will be equal to those bits:
    [
        111100,
        000000,
        100000,
        000000,
        110000,
        100000,
        111111,
        000,
        010,
        101,
        110,
        011,
        101,
        000
    ]
    """

    def __init__(self):
        self.game_rows = 6
        self.game_cols = 7
        self.bits_in_len = 3
        self.player_black = 1
        self.player_white = 0
        self.count_to_win = 4
        self.initial_state = self.encode_lists([[]] * self.game_cols)
        self.obs_shape = (2, self.game_rows, self.game_cols)

    @staticmethod
    def bits_to_int(bits):
        res = 0
        for b in bits:
            res *= 2
            res += b
        return res

    @staticmethod
    def int_to_bits(num, bits):
        res = []
        for _ in range(bits):
            res.append(num % 2)
            num //= 2
        return res[::-1]

    def encode_lists(self, field_lists):
        """
        Encode lists representation into the binary numbers
        :param field_lists: list of GAME_COLS lists with 0s and 1s
        :return: integer number with encoded game state
        """
        assert isinstance(field_lists, list)
        assert len(field_lists) == self.game_cols

        bits = []
        len_bits = []
        for col in field_lists:
            bits.extend(col)
            free_len = self.game_rows-len(col)
            bits.extend([0] * free_len)
            len_bits.extend(self.int_to_bits(free_len, bits=self.bits_in_len))
        bits.extend(len_bits)
        return self.bits_to_int(bits)

    def decode_binary(self, state_int):
        """
        Decode binary representation into the list view
        :param state_int: integer representing the field
        :return: list of GAME_COLS lists
        """
        assert isinstance(state_int, int)
        bits = self.int_to_bits(state_int,
                                bits=self.game_cols*self.game_rows + self.game_cols*self.bits_in_len)
        res = []
        len_bits = bits[self.game_cols*self.game_cols:]
        for col in range(self.game_cols):
            vals = bits[col*self.game_rows:(col+1)*self.game_rows]
            lens = self.bits_to_int(
                len_bits[col*self.bits_in_len:(col+1)*self.bits_in_len])
            if lens > 0:
                vals = vals[:-lens]
            res.append(vals)
        return res

    def possible_moves(self, state_int):
        """
        This function could be calculated directly from bits, but I'm too lazy
        :param state_int: field representation
        :return: the list of columns which we can make a move
        """
        assert isinstance(state_int, int)
        field = self.decode_binary(state_int)
        return [idx for idx, col in enumerate(field) if len(col) < self.game_rows]

    def _encode_list_state(self, dest_np, state_list, who_move):
        """
        In-place encodes list state into the zero numpy array
        :param dest_np: dest array, expected to be zero
        :param state_list: state of the game in the list form
        :param who_move: player index (game.PLAYER_WHITE or game.PLAYER_BLACK) who to move
        """
        assert dest_np.shape == self.obs_shape

        for col_idx, col in enumerate(state_list):
            for rev_row_idx, cell in enumerate(col):
                row_idx = self.game_rows - rev_row_idx - 1
                if cell == who_move:
                    dest_np[0, row_idx, col_idx] = 1.0
                else:
                    dest_np[1, row_idx, col_idx] = 1.0

    def state_lists_to_batch(self, state_lists, who_moves_lists):
        """
        Convert list of list states to batch for network
        :param state_lists: list of 'list states'
        :param who_moves_lists: list of player index who moves
        :return Variable with observations
        """
        assert isinstance(state_lists, list)
        batch_size = len(state_lists)
        batch = np.zeros((batch_size,) + self.obs_shape, dtype=np.float32)
        for idx, (state, who_move) in enumerate(zip(state_lists, who_moves_lists)):
            self._encode_list_state(batch[idx], state, who_move)
        return batch

    def _check_won(self, field, col, delta_row):
        """
        Check for horisontal/diagonal win condition for the last player moved in the column
        :param field: list of lists
        :param col: column index
        :param delta_row: if 0, checks for horisonal won, 1 for rising diagonal, -1 for falling
        :return: True if won, False if not
        """
        player = field[col][-1]
        coord = len(field[col])-1
        total = 1
        # negative dir
        cur_coord = coord - delta_row
        for c in range(col-1, -1, -1):
            if len(field[c]) <= cur_coord or cur_coord < 0 or cur_coord >= self.game_rows:
                break
            if field[c][cur_coord] != player:
                break
            total += 1
            if total == self.count_to_win:
                return True
            cur_coord -= delta_row
        # positive dir
        cur_coord = coord + delta_row
        for c in range(col+1, self.game_cols):
            if len(field[c]) <= cur_coord or cur_coord < 0 or cur_coord >= self.game_rows:
                break
            if field[c][cur_coord] != player:
                break
            total += 1
            if total == self.count_to_win:
                return True
            cur_coord += delta_row
        return False

    def move(self, state_int, col, player):
        """
        Perform move into given column. Assume the move could be performed, otherwise, assertion will be raised
        :param state_int: current state
        :param col: column to make a move
        :param player: player index (PLAYER_WHITE or PLAYER_BLACK
        :return: tuple of (state_new, won). Value won is bool, True if this move lead
        to victory or False otherwise (but it could be a draw)
        """
        assert isinstance(state_int, int)
        assert isinstance(col, int)
        assert 0 <= col < self.game_cols
        assert player == self.player_black or player == self.player_white
        field = self.decode_binary(state_int)
        assert len(field[col]) < self.game_rows
        field[col].append(player)
        # check for victory: the simplest vertical case
        suff = field[col][-self.count_to_win:]
        won = suff == [player] * self.count_to_win
        if not won:
            won = any((self._check_won(field, col, 0),
                      self._check_won(field, col, 1),
                      self._check_won(field, col, -1)))
        state_new = self.encode_lists(field)
        return state_new, won

    def render(self, state_int):
        state_list = self.decode_binary(state_int)
        data = [[' '] * self.game_cols for _ in range(self.game_rows)]
        for col_idx, col in enumerate(state_list):
            for rev_row_idx, cell in enumerate(col):
                row_idx = self.game_rows - rev_row_idx - 1
                data[row_idx][col_idx] = str(cell)
        return [''.join(row) for row in data]
