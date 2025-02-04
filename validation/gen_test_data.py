import copy

import torch

from mcts.brute_force_mcst_node import BruteMCSTNode
from self_play.game_data import GameData, GameDataset
from tic_tac_toe.board import TicTacToeBoard
from tic_tac_toe.player import TicTacToeBot
from utils import read_config

if __name__ == "__main__":
    # Setup.
    n_games = 100
    cfg = read_config("data_gen_cfg.yaml")  # Choose config script here.
    clean_board = TicTacToeBoard(TicTacToeBot("p1", "X"), TicTacToeBot("p2", "O"))
    game_dataset = GameDataset()

    # Simulate full games.
    for game in range(n_games):
        print(f"Simulating game {game} of {n_games - 1}")
        game_board = copy.deepcopy(clean_board)
        game_data = GameData()
        while not game_board.game_result:
            turn_node = BruteMCSTNode(None, game_board, cfg)
            actions, probs = turn_node.search()
            best_action = actions[probs.argmax()]
            game_board.exec_move(tuple(best_action))
            game_data.append_turn(game_board, probs, game_board.last_player, game_board.last_move)
        game_data.finalise(game_board.last_player, game_board.game_result, cfg["mcts"]["rewards"])
        game_dataset.append_game(game_data)

    # Save as torch object.
    torch.save(game_dataset, f"validation/data/game_dataset_{n_games}_games_negated.pt")
