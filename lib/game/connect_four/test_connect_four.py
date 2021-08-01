import pytest
from lib.game.connect_four import connect_four


@pytest.fixture
def game():
    return connect_four.ConnectFour()


class TestBits:
    def test_bits_to_int(self, game):
        v = game.bits_to_int([1, 0, 0])
        assert v == 4
        v = game.bits_to_int([])
        assert v == 0

    def test_int_to_bits(self, game):
        v = game.int_to_bits(1, bits=1)
        assert v == [1]
        v = game.int_to_bits(1, bits=5)
        assert v == [0, 0, 0, 0, 1]
        v = game.int_to_bits(5, bits=7)
        assert v == [0, 0, 0, 0, 1, 0, 1]


class TestGameEncoding:
    def test_simple_encode(self, game):
        e = game.encode_lists([[]]*7)
        assert e == 0b000000000000000000000000000000000000000000110110110110110110110
        e = game.encode_lists([[1]*6]*7)
        assert e == 0b111111111111111111111111111111111111111111000000000000000000000
        e = game.encode_lists([[0]*6]*7)
        assert e == 0

    def test_simple_decode(self, game):
        g = game.decode_binary(
            0b000000000000000000000000000000000000000000110110110110110110110)
        assert g == [[]]*7
        g = game.decode_binary(
            0b111111111111111111111111111111111111111111000000000000000000000)
        assert g == [[1]*6]*7
        g = game.decode_binary(0)
        assert g == [[0]*6]*7


class TestMoveFunctions:
    def test_possible_moves(self, game):
        r = game.possible_moves(0)
        assert r == []
        r = game.possible_moves(
            0b111111111111111111111111111111111111111111000000000000000000000)
        assert r == []
        r = game.possible_moves(
            0b000000000000000000000000000000000000000000110110110110110110110)
        assert r == [0, 1, 2, 3, 4, 5, 6]

    def test_move_vertical_win(self, game):
        f = game.encode_lists([[]]*7)

        f, won = game.move(f, 0, 1)
        assert won == False
        l = game.decode_binary(f)
        assert l == [[1]] + [[]]*6

        f, won = game.move(f, 0, 1)
        assert won == False
        l = game.decode_binary(f)
        assert l == [[1, 1]] + [[]]*6

        f, won = game.move(f, 0, 1)
        assert won == False
        l = game.decode_binary(f)
        assert l == [[1, 1, 1]] + [[]]*6

        f, won = game.move(f, 0, 1)
        assert won == True
        l = game.decode_binary(f)
        assert l == [[1, 1, 1, 1]] + [[]]*6

    def test_move_horizontal_win(self, game):
        f = game.encode_lists([[]]*7)

        f, won = game.move(f, 0, 1)
        assert won == False
        l = game.decode_binary(f)
        assert l == [[1]] + [[]]*6

        f, won = game.move(f, 1, 1)
        assert won == False
        l = game.decode_binary(f)
        assert l == [[1], [1]] + [[]]*5

        f, won = game.move(f, 3, 1)
        assert won == False
        l = game.decode_binary(f)
        assert l == [[1], [1], [], [1], [], [], []]

        f, won = game.move(f, 2, 1)
        assert won == True
        l = game.decode_binary(f)
        assert l == [[1], [1], [1], [1], [], [], []]

    def test_move_diags(self, game):
        f = game.encode_lists([
            [0, 0, 0, 1],
            [0, 0, 1],
            [0],
            [1],
            [], [], []
        ])
        _, won = game.move(f, 2, 1)
        assert won == True
        _, won = game.move(f, 2, 0)
        assert won == False

        f = game.encode_lists([
            [],
            [0, 1],
            [0, 0, 1],
            [1, 0, 0, 1],
            [], [], []
        ])
        _, won = game.move(f, 0, 1)
        assert won == True
        _, won = game.move(f, 0, 0)
        assert won == False

    def test_tricky(self, game):
        f = game.encode_lists([
            [0, 1, 1],
            [1, 0],
            [0, 1],
            [0, 0, 1],
            [0, 0],
            [1, 1, 1, 0],
            []
        ])
        s, won = game.move(f, 4, 0)
        assert won == True
        assert s == 3531389463375529686
