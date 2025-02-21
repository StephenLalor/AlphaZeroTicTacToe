import timeit

import mlflow

from mcts.mcst_node import batch_search, search
from neural_networks.load_model import load_mdl_for_inference
from tic_tac_toe.board import TicTacToeBoard
from tic_tac_toe.player import TicTacToeBot
from utils import read_config

if __name__ == "__main__":
    cfg = read_config("training_cfg.yaml")
    mlflow.set_tracking_uri("http://localhost:5000")
    model = load_mdl_for_inference("test_large", 12)
    model.eval()

    N_BOARDS = 300
    N_RUNS = 1
    boards = [
        TicTacToeBoard(TicTacToeBot("p1", "X"), TicTacToeBot("p2", "O")) for _ in range(N_BOARDS)
    ]

    def non_batch_search(boards, model, cfg):
        for board in boards:
            search(board, model, cfg)

    print(f"------- Performance for {N_BOARDS} boards over {N_RUNS} runs -------")

    # 1. Batch.
    batch_time = timeit.timeit(lambda: batch_search(boards, model, cfg), number=N_RUNS)
    print(f"Batch: {batch_time:.4f}s ({batch_time / N_RUNS:.4f}s/run)")

    # 2. Non-batch.
    non_batch_time = timeit.timeit(lambda: non_batch_search(boards, model, cfg), number=N_RUNS)
    print(f"Non Batch: {non_batch_time:.4f}s ({non_batch_time / N_RUNS:.4f}s/run)")
