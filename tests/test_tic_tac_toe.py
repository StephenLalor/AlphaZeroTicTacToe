import copy

import pytest

from tic_tac_toe.board import TicTacToeBoard, TicTacToeException
from tic_tac_toe.player import TicTacToeBot


@pytest.fixture
def clean_board():
    return copy.deepcopy(TicTacToeBoard(TicTacToeBot("p1", "X"), TicTacToeBot("p2", "O")))


def test_repeat_move_validation(clean_board: TicTacToeBoard):
    with pytest.raises(TicTacToeException, match=".* is not valid move for this board."):
        clean_board.exec_move((1, 1))
        clean_board.exec_move((1, 1))


def test_oob_move_validation(clean_board: TicTacToeBoard):
    with pytest.raises(TicTacToeException, match=".* is not valid move for this board."):
        clean_board.exec_move((-2, 93))


def test_terminated_move_validation(clean_board: TicTacToeBoard):
    clean_board.exec_move((0, 0))
    clean_board.exec_move((2, 2))
    clean_board.exec_move((0, 1))
    clean_board.exec_move((1, 1))
    clean_board.exec_move((0, 2))
    with pytest.raises(TicTacToeException, match="Game already ended."):
        clean_board.exec_move((1, 2))


def test_row_win_1(clean_board: TicTacToeBoard):
    clean_board.exec_move((0, 0))  # X
    clean_board.exec_move((2, 2))  # O
    clean_board.exec_move((0, 1))  # X
    clean_board.exec_move((1, 1))  # O
    clean_board.exec_move((0, 2))  # X <-- X wins
    assert clean_board.game_result == "win"
    assert clean_board.last_move == (0, 2)


def test_row_win_2(clean_board: TicTacToeBoard):
    clean_board.exec_move((0, 0))  # X
    clean_board.exec_move((1, 0))  # O
    clean_board.exec_move((0, 1))  # X
    clean_board.exec_move((1, 1))  # O
    clean_board.exec_move((2, 0))  # X
    clean_board.exec_move((1, 2))  # O  <-- O wins
    assert clean_board.game_result == "win"
    assert clean_board.last_move == (1, 2)


def test_diag_win_1(clean_board: TicTacToeBoard):
    clean_board.exec_move((0, 0))  # X
    clean_board.exec_move((0, 1))  # O
    clean_board.exec_move((1, 1))  # X
    clean_board.exec_move((0, 2))  # O
    clean_board.exec_move((2, 2))  # X  <-- X wins
    assert clean_board.game_result == "win"
    assert clean_board.last_move == (2, 2)


def test_diag_win_2(clean_board: TicTacToeBoard):
    clean_board.exec_move((0, 0))  # X
    clean_board.exec_move((0, 2))  # O
    clean_board.exec_move((0, 1))  # X
    clean_board.exec_move((1, 1))  # O
    clean_board.exec_move((2, 1))  # X
    clean_board.exec_move((2, 0))  # O  <-- O wins
    assert clean_board.game_result == "win"
    assert clean_board.last_move == (2, 0)


def test_draw(clean_board: TicTacToeBoard):
    clean_board.exec_move((0, 0))  # X
    clean_board.exec_move((0, 1))  # O
    clean_board.exec_move((0, 2))  # X
    clean_board.exec_move((1, 0))  # O
    clean_board.exec_move((1, 2))  # X
    clean_board.exec_move((1, 1))  # O
    clean_board.exec_move((2, 0))  # X
    clean_board.exec_move((2, 2))  # O
    clean_board.exec_move((2, 1))  # X <-- Board Full
    assert clean_board.game_result == "draw"
    assert clean_board.last_move == (2, 1)
