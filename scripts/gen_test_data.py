import torch

from tic_tac_toe.board import TicTacToeBoard
from tic_tac_toe.player import TicTacToeBot
from utils import read_config
from validation.gen_test_game import gen_data_multi_proc_batch

if __name__ == "__main__":
    # Setup.
    cfg = read_config("data_gen_cfg.yaml")  # Choose config script here.
    clean_board = TicTacToeBoard(TicTacToeBot("p1", "X"), TicTacToeBot("p2", "O"))
    game_dataset = gen_data_multi_proc_batch(clean_board, cfg)

    # Save as torch object.
    torch.save(game_dataset, f"validation/data/game_dataset_{cfg['perf']['n_games']}_games.pt")
