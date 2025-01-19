import concurrent as conc
import copy
import logging
import multiprocessing as mp

import mlflow
import numpy as np
import torch
from torch.nn.functional import cross_entropy, mse_loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchinfo import summary

from mcts.mcst_node import MCSTNode
from neural_networks.data_prep import get_input_feats
from neural_networks.tic_tac_toe_net import TicTacToeNet
from self_play.game_data import GameData, GameDataset
from tic_tac_toe.board import TicTacToeBoard
from tic_tac_toe.player import TicTacToeBot

logger = logging.getLogger("myapp.module")


class TicTacToeTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.workers = mp.cpu_count()
        # Set up initial clean game state.
        self.clean_board = TicTacToeBoard(TicTacToeBot("p1", "X"), TicTacToeBot("p2", "O"))
        # Set up NN and optimiser parameter groups.
        self.model = TicTacToeNet(get_input_feats(self.clean_board), cfg["nn"])
        self.optimiser = Adam(
            [
                {"params": self.model.get_pol_param_grp(), "lr": cfg["nn"]["pol_lr"]},
                {"params": self.model.get_val_param_grp(), "lr": cfg["nn"]["val_lr"]},
                {"params": self.model.get_oth_param_grp(), "lr": cfg["nn"]["oth_lr"]},
            ]
        )
        # MLflow logging.
        self.tot_steps = 0  # Current training step count.
        self.tot_epochs = 0  # Current training epoch count.

    def self_play(self, model: TicTacToeNet, clean_board: TicTacToeBoard, cfg: dict) -> GameData:
        # Board, action probabilities and reward for each turn.
        game_data = GameData()
        # Concurrent processes need their own RNG object.
        rng = np.random.default_rng()  # For stochastic selection.
        # Play until termination.
        board = copy.deepcopy(clean_board)
        while not board.game_result:
            # Begin search from the current position.
            mct = MCSTNode(None, None, board, cfg["mcts"])
            actions, probs = mct.search(model)
            # Do stochastic move selection and execute.
            action = rng.choice(actions, p=probs)
            board.exec_move(tuple(action))
            # Update history.
            game_data.append_turn(board, probs, board.last_player)
        # Assign reward for each turn now that result is known.
        game_data.finalise(board.last_player, board.game_result, cfg["mcts"]["rewards"])
        return game_data

    def train(self, loader: DataLoader):
        for batch, (batch_states, batch_pol_targets, batch_rewards) in enumerate(loader):
            # Forward pass predict policy and value.
            pred_policy, pred_value = self.model(batch_states)  # TODO: move states to device?
            pred_policy = pred_policy.to(self.device)
            pred_value = pred_value.to(self.device)
            # Compute loss vs self play results.
            pol_loss = cross_entropy(pred_policy, batch_pol_targets)
            val_loss = mse_loss(pred_value.squeeze(), batch_rewards)
            tot_loss = pol_loss + val_loss
            # Optimise.
            self.optimiser.zero_grad()  # Reset for new sample.
            tot_loss.backward()
            self.optimiser.step()  # Update gradients.

            # Log loss metrics every log_period batches.
            if batch % self.cfg["exp"]["log_period"] == 0:
                mlflow.log_metric("tot_loss", f"{tot_loss.item():4f}", step=self.tot_steps)
                mlflow.log_metric("pol_loss", f"{pol_loss.item():4f}", step=self.tot_steps)
                mlflow.log_metric("val_loss", f"{val_loss.item():4f}", step=self.tot_steps)
            self.tot_steps += 1

    def evaluate(self, loader: DataLoader):
        # Evaluate model on playout data by accumulating loss.
        pol_loss, val_loss, tot_loss = 0, 0, 0
        with torch.no_grad():
            for batch_states, batch_pol_targets, batch_rewards in loader:
                pred_policy, pred_value = self.model(batch_states)
                pol_loss += cross_entropy(pred_policy, batch_pol_targets)
                val_loss += mse_loss(pred_value.squeeze(), batch_rewards)

        # Average the loss values and log.
        tot_loss += pol_loss + val_loss
        tot_loss = tot_loss / len(loader)
        mlflow.log_metric("epoch_tot_loss", f"{tot_loss:4f}", step=self.tot_epochs)
        pol_loss = pol_loss / len(loader)
        mlflow.log_metric("epoch_pol_loss", f"{pol_loss:4f}", step=self.tot_epochs)
        val_loss = val_loss / len(loader)
        mlflow.log_metric("epoch_val_loss", f"{val_loss:4f}", step=self.tot_epochs)

    def self_train(self):
        # Initialise mlflow logging.
        mlflow.set_tracking_uri(self.cfg["exp"]["uri"])
        mlflow.set_experiment(self.cfg["exp"]["name"])
        mlflow.start_run()
        with open("model_summary.txt", "w", encoding="utf-8") as f:
            f.write(str(summary(self.model)))
        mlflow.log_artifact("model_summary.txt")
        mlflow.log_params(self.cfg)

        # Begin self play and training cycle.
        for cycle in range(self.cfg["self_play"]["cycles"]):
            # Generate many games with self play.
            print(f"--------- Cycle {cycle} ---------")
            game_data = GameDataset()
            self.model.eval()  # Not training yet.
            # Play out games concurrently.
            with conc.futures.ProcessPoolExecutor(max_workers=self.workers) as pool:
                futures = []
                for run in range(self.cfg["self_play"]["playouts"]):
                    futures.append(
                        pool.submit(self.self_play, self.model, self.clean_board, self.cfg)
                    )
                for future in conc.futures.as_completed(futures):
                    game_data.append_game(future.result())

            # Begin training using self play data.
            game_data_loader = game_data.to_loader(self.cfg["nn"]["batch_size"])
            for epoch in range(self.cfg["self_play"]["epochs"]):
                print(f"\tEpoch {epoch}")
                # Train.
                self.model.train()
                self.train(game_data_loader)
                # Evaluate.
                self.model.eval()
                self.evaluate(game_data_loader)
                self.tot_epochs += 1

            # Save model to MLflow.
            mlflow.pytorch.log_model(self.model, f"model_cycle_{str(cycle)}")
        mlflow.end_run()
