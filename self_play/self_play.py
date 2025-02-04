# TODO: Docstrings!
import copy
import logging

import mlflow
import numpy as np
import torch
from torch.nn.functional import cross_entropy, mse_loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchinfo import summary

from mcts.mcst_node import search
from neural_networks.tic_tac_toe_net import TicTacToeNet
from self_play.game_data import GameData, GameDataset
from tic_tac_toe.board import TicTacToeBoard
from tic_tac_toe.player import TicTacToeBot

logger = logging.getLogger("myapp.module")


class TicTacToeTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.rng = np.random.default_rng()
        # Set up initial clean game state.
        self.clean_board = TicTacToeBoard(TicTacToeBot("p1", "X"), TicTacToeBot("p2", "O"))
        # Set up NN and optimiser parameter groups.
        self.model = TicTacToeNet(cfg["nn"])
        self.optimiser = Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001, eps=1e-8)
        # Test data for evaluation.
        test_data = torch.load(cfg["exp"]["test_data_path"], weights_only=False)
        self.test_data: DataLoader = test_data.to_loader(len(test_data))  # One large batch.
        # MLflow logging.
        self.tot_steps = 0  # Current training step count.

    def self_play(self, model: TicTacToeNet, clean_board: TicTacToeBoard, cfg: dict) -> GameData:
        # Board, action probabilities and reward for each turn.
        game_data = GameData()
        # Play until termination.
        board = copy.deepcopy(clean_board)
        while not board.game_result:
            # Begin search from the current position.
            actions, probs = search(board, model, cfg)
            # Do stochastic move selection and execute.
            probs = probs ** (1 / cfg["mcts"]["temperature"])
            probs = probs / probs.sum()
            action = self.rng.choice(actions, p=probs)
            board.exec_move(tuple(action))
            # Update history.
            game_data.append_turn(board, probs, board.last_player, board.last_move)
        # Assign reward for each turn now that result is known.
        game_data.finalise(board.last_player, board.game_result, cfg["mcts"]["rewards"])
        return game_data

    def track_grad_norms(self):
        # Grad norm from conv block.
        conv_grad_norm = torch.norm(self.model.conv_block.conv.weight.grad).item()
        mlflow.log_metric("grad_norm/conv", conv_grad_norm, self.tot_steps)
        # Grad norm from final res block.
        i = self.cfg["nn"]["res_blocks"] - 1  # Index of last.
        res_grad_norm = torch.norm(self.model.res_blocks[i].batch_norm_2.weight.grad).item()
        mlflow.log_metric("grad_norm/res", res_grad_norm, self.tot_steps)
        # Grad norm from final value head layer.
        value_grad_norm = torch.norm(self.model.value_head.linear_3.weight.grad).item()
        mlflow.log_metric("grad_norm/val", value_grad_norm, self.tot_steps)
        # Grad norm from final policy head layer.
        policy_grad_norm = torch.norm(self.model.policy_head.linear_2.weight.grad).item()
        mlflow.log_metric("grad_norm/pol", policy_grad_norm, self.tot_steps)

    def train(self, loader: DataLoader):
        for batch, (batch_states, batch_pol_targets, batch_rewards) in enumerate(loader):
            # Target tensors to device.
            batch_states = batch_states.to(self.model.device)
            batch_pol_targets = batch_pol_targets.to(self.model.device)
            batch_rewards = batch_rewards.to(self.model.device)
            # Forward pass predict policy and value.
            pred_policy, pred_value = self.model(batch_states)
            # Compute loss vs self play results.
            pol_loss = cross_entropy(pred_policy, batch_pol_targets)
            val_loss = mse_loss(pred_value, batch_rewards)
            tot_loss = pol_loss + val_loss
            # Optimise.
            self.optimiser.zero_grad()
            tot_loss.backward()
            if not self.tot_steps % self.cfg["exp"]["log_period"]:
                self.track_grad_norms()
            self.optimiser.step()
            # Update tracking.
            self.tot_steps += 1

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, log_as: str):
        for i, (test_states, test_pol_targets, test_rewards) in enumerate(loader):
            # Target tensors to device.
            test_states = test_states.to(self.model.device)
            test_pol_targets = test_pol_targets.to(self.model.device)
            test_rewards = test_rewards.to(self.model.device)
            # Predict on test data.
            pred_policy, pred_value = self.model(test_states)
            # Compute loss vs self play results.
            pol_loss = cross_entropy(pred_policy, test_pol_targets)
            val_loss = mse_loss(pred_value, test_rewards)
            tot_loss = pol_loss + val_loss
            # Record loss.
            mlflow.log_metric(f"{log_as}_loss/tot", round(tot_loss.item(), 2), self.tot_steps)
            mlflow.log_metric(f"{log_as}_loss/pol", round(pol_loss.item(), 2), self.tot_steps)
            mlflow.log_metric(f"{log_as}_loss/val", round(val_loss.item(), 2), self.tot_steps)

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
            game_dataset = GameDataset()
            self.model.eval()  # Not training yet.
            # Play out games concurrently.
            for run in range(self.cfg["self_play"]["playouts"]):
                game_result = self.self_play(self.model, self.clean_board, self.cfg)
                game_dataset.append_game(game_result)

            # Begin training using self play data.
            game_data_loader = game_dataset.to_loader(self.cfg["nn"]["batch_size"])
            for epoch in range(self.cfg["self_play"]["epochs"]):
                # Train.
                self.model.train()
                self.train(game_data_loader)

            # Evaluate against training and test data.
            self.model.eval()
            self.evaluate(game_data_loader, "cycle")
            self.evaluate(self.test_data, "test")

            # Save model to MLflow.
            mlflow.pytorch.log_model(self.model, f"model_cycle_{str(cycle)}")
        mlflow.end_run()
