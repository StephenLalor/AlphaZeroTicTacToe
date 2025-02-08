import copy

import numpy as np
import pytest
from numpy.random import SeedSequence

from mcts.brute_force_mcst_node import BruteMCSTNode
from tic_tac_toe.board import TicTacToeBoard
from tic_tac_toe.player import TicTacToeBot
from utils import read_config


@pytest.fixture
def clean_board():
    return copy.deepcopy(TicTacToeBoard(TicTacToeBot("p1", "X"), TicTacToeBot("p2", "O")))


@pytest.fixture
def cfg():
    return copy.deepcopy(read_config("test_training_cfg.yaml"))


@pytest.fixture
def seed_seq():
    return SeedSequence()


@pytest.mark.repeat(3)
def test_best_move_high_sim_1(clean_board, cfg, seed_seq):
    """
    Test best move is chosen using high number of simulations.
    """
    # Create board with obvious best move.
    clean_board.exec_move((0, 0))  # X
    clean_board.exec_move((2, 2))  # O
    clean_board.exec_move((0, 1))  # X
    clean_board.exec_move((1, 1))  # O
    # Do simulation.
    cfg["mcts"]["sim_lim"] = 100  # Ensure simulation count is always high.
    node = BruteMCSTNode(None, clean_board, cfg, seed_seq)
    actions, probs = node.search()
    # Check maximum probability move is correct.
    assert (actions[probs.argmax()] == np.array([0, 2])).all()


@pytest.mark.repeat(3)
def test_best_move_high_sim_2(clean_board, cfg, seed_seq):
    """
    Test best move is chosen using high number of simulations.
    """
    # Create board with obvious best move.
    clean_board.exec_move((0, 0))  # X
    clean_board.exec_move((1, 0))  # O
    clean_board.exec_move((0, 1))  # X
    clean_board.exec_move((1, 1))  # O
    clean_board.exec_move((2, 0))  # X
    # Do simulation.
    cfg["mcts"]["sim_lim"] = 100  # Ensure simulation count is always high.
    node = BruteMCSTNode(None, clean_board, cfg, seed_seq)
    actions, probs = node.search()
    # Check maximum probability move is correct.
    assert (actions[probs.argmax()] == np.array([1, 2])).all()


@pytest.mark.repeat(3)
def test_best_move_high_sim_3(clean_board, cfg, seed_seq):
    """
    Test best move is chosen using high number of simulations.
    """
    # Create board with obvious best move.
    clean_board.exec_move((0, 0))  # X
    clean_board.exec_move((0, 1))  # O
    clean_board.exec_move((1, 1))  # X
    clean_board.exec_move((0, 2))  # O
    # Do simulation.
    cfg["mcts"]["sim_lim"] = 100  # Ensure simulation count is always high.
    node = BruteMCSTNode(None, clean_board, cfg, seed_seq)
    actions, probs = node.search()
    # Check maximum probability move is correct.
    assert (actions[probs.argmax()] == np.array([2, 2])).all()


@pytest.mark.repeat(3)
def test_best_move_high_sim_4(clean_board, cfg, seed_seq):
    """
    Test best move is chosen using high number of simulations.
    """
    # Create board with obvious best move.
    clean_board.exec_move((0, 0))  # X
    clean_board.exec_move((0, 1))  # O
    clean_board.exec_move((0, 2))  # X
    clean_board.exec_move((1, 0))  # O
    clean_board.exec_move((1, 1))  # X
    clean_board.exec_move((2, 0))  # O
    clean_board.exec_move((2, 1))  # X
    # Do simulation.
    cfg["mcts"]["sim_lim"] = 100  # Ensure simulation count is always high.
    node = BruteMCSTNode(None, clean_board, cfg, seed_seq)
    actions, probs = node.search()
    # Check maximum probability move is correct.
    assert (actions[probs.argmax()] == np.array([2, 2])).all()
