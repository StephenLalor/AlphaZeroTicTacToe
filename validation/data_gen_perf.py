import timeit

from tic_tac_toe.board import TicTacToeBoard
from tic_tac_toe.player import TicTacToeBot
from utils import read_config
from validation.gen_test_game import (
    gen_data_multi_proc,
    gen_data_multi_proc_batch,
    gen_data_single_proc,
)

if __name__ == "__main__":
    clean_board = TicTacToeBoard(TicTacToeBot("p1", "X"), TicTacToeBot("p2", "O"))
    cfg = read_config("data_gen_cfg.yaml")
    n_runs = cfg["perf"]["n_runs"]
    print(f"------- Performance for {cfg['perf']['n_games']} games over {n_runs} runs -------")

    # 1. Multi processing
    multi_time = timeit.timeit(lambda: gen_data_multi_proc(clean_board, cfg), number=n_runs)
    print(f"Multi Proc Gen: {multi_time:.4f}s ({multi_time / n_runs:.4f}s/run)")

    # 2. Naive Generation
    naive_time = timeit.timeit(lambda: gen_data_single_proc(clean_board, cfg), number=n_runs)
    print(f"Naive Gen: {naive_time:.4f}s ({naive_time / n_runs:.4f}s/run)")

    # 2. Multi batch processing
    multi_batch_time = timeit.timeit(
        lambda: gen_data_multi_proc_batch(clean_board, cfg), number=n_runs
    )
    print(f"Multi Proc Batch Gen: {multi_batch_time:.4f}s ({multi_batch_time / n_runs:.4f}s/run)")
