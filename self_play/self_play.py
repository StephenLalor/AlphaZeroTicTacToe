import copy
import logging

import numpy as np
import torch
from torch.nn.functional import cross_entropy, mse_loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mcts.alpha_zero_mcts import SmartMCSTNode
from neural_networks.data_prep import get_input_feats
from neural_networks.tic_tac_toe_net import TicTacToeNet
from self_play.game_data import GameData, GameDataset, collate_game_data
from tic_tac_toe.board import TicTacToeBoard
from tic_tac_toe.player import TicTacToeBot

logger = logging.getLogger("myapp.module")


class TicTacToeTrainer:
    def __init__(self, cfg: dict, path: str):
        self.cfg = cfg
        self.path = path
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
        # Set up Tensorboard.
        self.writer = SummaryWriter(path)
        self.total_steps = 0

    def self_play(self) -> GameData:
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

    def train(self, training_data: GameDataset):
        # Set up batch loader with shuffling.
        training_data.to(self.device)
        batch_size = self.cfg["nn"]["batch_size"]
        train_loader = DataLoader(training_data, batch_size, True, collate_fn=collate_game_data)
        val_loss_hist, pol_loss_hist, tot_loss_hist = [], [], []

        # Training loop.
        for i, (batch_states, batch_pol_targets, batch_rewards) in enumerate(train_loader):
            print(f"Training loop iteration {i}")
            # Forward pass predict policy and value.
            pred_policy, pred_value = self.model(batch_states)
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
            self.total_steps += 1

            # Log total training loss.
            tot_loss_hist.append(tot_loss.item())
            self.writer.add_scalar("Loss/TotalLoss", tot_loss.item(), self.total_steps)
            # Log policy training loss.
            pol_loss_hist.append(pol_loss.item())
            self.writer.add_scalar("Loss/PolicyLoss", pol_loss.item(), self.total_steps)
            # Log value training loss.
            val_loss_hist.append(val_loss.item())
            self.writer.add_scalar("Loss/ValueLoss", val_loss.item(), self.total_steps)

        return np.array(val_loss_hist), np.array(pol_loss_hist), np.array(tot_loss_hist)

    def learn(self):
        for cycle in range(self.cfg["self_play"]["cycles"]):
            # Begin self play cycle.
            print(f"--------- Cycle {cycle} ---------")
            training_data = GameDataset()
            self.model.eval()  # Eval mode as not training yet.
            for playout in range(self.cfg["self_play"]["playouts"]):
                game = self.self_play()
                training_data.append_game(game)

            # Begin training using self play data.
            self.model.train()  # Training mode as we now will train.
            for epoch in range(self.cfg["self_play"]["epochs"]):
                val_loss_hist, pol_loss_hist, tot_loss_hist = self.train(training_data)
                # Aggregate metrics for this epoch.
                metrics = {
                    "tot_loss": tot_loss_hist.mean().item(),
                    "pol_loss": pol_loss_hist.mean().item(),
                    "val_loss": val_loss_hist.mean().item(),
                }
                # Log epoch loss metrics.
                self.writer.add_scalar("Epoch/TotalLoss", metrics["tot_loss"], self.total_steps)
                self.writer.add_scalar("Epoch/PolicyLoss", metrics["pol_loss"], self.total_steps)
                self.writer.add_scalar("Epoch/ValueLoss", metrics["val_loss"], self.total_steps)

            # Save model and add to tensorboard.
            torch.save(self.model.state_dict(), f"{self.path}/model_{str(cycle)}.pt")
            torch.save(self.optimiser.state_dict(), f"{self.path}/optimiser_{str(cycle)}.pt")

        # Log hyperparameters with final epoch loss metrics.
        # self.writer.add_hparams(flatten_dict(self.cfg), metrics)  # NOTE: add_hparams() is buggy.
        self.writer.add_graph(self.model, get_input_feats(self.clean_board))
        self.writer.close()
