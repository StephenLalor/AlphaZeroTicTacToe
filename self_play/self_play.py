import copy
import logging

import mlflow
import numpy as np
import torch
from torch.nn.functional import cross_entropy, mse_loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchinfo import summary

from mcts.alpha_zero_mcts import SmartMCSTNode
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
        # Set up initial clean game state.
        self.clean_board = TicTacToeBoard(
            TicTacToeBot("p1", 1, "random"), TicTacToeBot("p2", 2, "random")
        )
        # Set up NN and optimiser parameter groups.
        self.model = TicTacToeNet(get_input_feats(self.clean_board), cfg["nn"])
        self.optimiser = Adam(
            [
                {"params": self.model.get_pol_param_grp(), "lr": cfg["nn"]["pol_lr"]},
                {"params": self.model.get_val_param_grp(), "lr": cfg["nn"]["val_lr"]},
                {"params": self.model.get_oth_param_grp(), "lr": cfg["nn"]["oth_lr"]},
            ]
        )
        self.rng = np.random.default_rng()  # For stochastic selection.
        self.tot_steps = 0  # Current training step count.
        self.tot_epochs = 0  # Current training epoch count.

    def self_play(self) -> GameData:
        # TODO: Do I need to use torch.no_grad() here as well?
        # Board, action probabilities and reward for each turn.
        game_data = GameData()
        # Play until termination.
        board = copy.deepcopy(self.clean_board)
        while not board.game_result:
            # Begin search from the current position.
            mct = SmartMCSTNode(None, None, board, self.model, self.cfg["mcts"])
            actions, probs = mct.run_search()
            # Do stochastic move selection and execute.
            action = self.rng.choice(actions, p=probs)
            board.exec_move(tuple(action))
            # Update history.
            game_data.append_turn(board, probs, board.last_player)

        # Assign reward for each turn now that result is known.
        game_data.finalise(board.last_player, board.game_result, self.cfg["mcts"]["rewards"])
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
        with mlflow.start_run():
            # Initial logging.
            # TODO: Write as tempfile instead.
            with open("model_summary.txt", "w", encoding="utf-8") as f:
                f.write(str(summary(self.model)))
            mlflow.log_artifact("model_summary.txt")
            mlflow.log_params(self.cfg)

            for cycle in range(self.cfg["self_play"]["cycles"]):
                # Begin self play cycle.
                print(f"--------- Cycle {cycle} ---------")
                game_data = GameDataset()
                self.model.eval()  # Eval mode as not training yet.
                for playout in range(self.cfg["self_play"]["playouts"]):
                    game = self.self_play()
                    game_data.append_game(game)

                # Begin training using self play data.
                game_data_loader = game_data.to_loader(self.cfg["nn"]["batch_size"])
                for epoch in range(self.cfg["self_play"]["epochs"]):
                    print(f"\t--------- Epoch {epoch} ---------")
                    # Train.
                    self.model.train()
                    self.train(game_data_loader)
                    # Evaluate.
                    self.model.eval()
                    self.evaluate(game_data_loader)
                    self.tot_epochs += 1

                # Save model to MLflow.
                mlflow.pytorch.log_model(self.model, f"model_cycle_{str(cycle)}")
