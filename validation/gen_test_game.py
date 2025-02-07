import copy
from concurrent import futures
from multiprocessing import cpu_count

from numpy.random import SeedSequence

from mcts.brute_force_mcst_node import BruteMCSTNode
from self_play.game_data import GameData, GameDataset
from tic_tac_toe.board import TicTacToeBoard


def _gen_game(clean_board: TicTacToeBoard, cfg: dict, seed: SeedSequence):
    game_data = GameData()
    game_board = copy.deepcopy(clean_board)
    while not game_board.game_result:
        # Search to find best move.
        turn_board = copy.deepcopy(game_board)  # Board at start of turn.
        turn_node = BruteMCSTNode(None, game_board, cfg, seed)
        actions, probs = turn_node.search()
        # Execute best move.
        best_action = actions[probs.argmax()]
        game_board.exec_move(tuple(best_action))
        # Add data to game history.
        game_data.append_turn(turn_board, probs, turn_board.last_player, turn_board.last_move)
    game_data.finalise(game_board.last_player, game_board.game_result, cfg["mcts"]["rewards"])
    return game_data


def gen_data_single_proc(clean_board: TicTacToeBoard, cfg: dict):
    game_dataset = GameDataset()
    master_seeds = SeedSequence().spawn(cfg["perf"]["n_games"])
    for i in range(cfg["perf"]["n_games"]):
        game_data = _gen_game(clean_board, cfg, master_seeds[i])
        game_dataset.append_game(game_data)
    return game_dataset


def gen_data_multi_proc(clean_board: TicTacToeBoard, cfg: dict):
    game_dataset = GameDataset()
    future_games = []
    n_workers = cpu_count()
    master_seeds = SeedSequence().spawn(cfg["perf"]["n_games"])
    with futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        # Queue up future games for parallel execution.
        for i in range(cfg["perf"]["n_games"]):
            future_games.append(pool.submit(_gen_game, clean_board, cfg, master_seeds[i]))
        # Execute games and add to game data set as they finish.
        for future in futures.as_completed(future_games):
            game_dataset.append_game(future.result())
    return game_dataset


def _batch_worker(clean_board: TicTacToeBoard, cfg: dict, batch_size: int, seeds: SeedSequence):
    games = []
    for i in range(batch_size):
        games.append(_gen_game(clean_board, cfg, seeds[i]))
    return games


def _get_batch_size(n_games: int, n_workers: int):
    batch_size, remaining = divmod(n_games, n_workers)
    for _ in range(n_workers):
        adj_batch_size = batch_size
        if remaining > 0:
            adj_batch_size += 1
            remaining -= 1
        yield adj_batch_size


def gen_data_multi_proc_batch(clean_board: TicTacToeBoard, cfg: dict):
    # Execute games as batches for each worker.
    game_dataset = GameDataset()
    n_workers = cpu_count()
    future_games = []
    seed_seq = SeedSequence()
    batch_size_gen = _get_batch_size(cfg["perf"]["n_games"], n_workers)
    with futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        # Assign batch workers to execute all games.
        for adj_batch_size in batch_size_gen:
            worker_seeds = seed_seq.spawn(adj_batch_size)
            job = pool.submit(_batch_worker, clean_board, cfg, adj_batch_size, worker_seeds)
            future_games.append(job)
        # Execute games.
        for future in futures.as_completed(future_games):
            # Append each to GameDataSet.
            games_list = future.result()
            for game in games_list:
                game_dataset.append_game(game)
    return game_dataset
