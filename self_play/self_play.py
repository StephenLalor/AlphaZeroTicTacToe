import copy
import logging

import numpy as np
import torch
from torch.nn.functional import cross_entropy, mse_loss
from torch.optim import AdamW
from torch.utils.data import DataLoader

from mcts.alpha_zero_mcts import SmartMCSTNode
from neural_networks.data_prep import get_input_feats
from neural_networks.tic_tac_toe_net import TicTacToeNet
from self_play.game_data import GameData, GameDataset, collate_game_data
from tic_tac_toe.board import TicTacToeBoard
from tic_tac_toe.player import TicTacToeBot

logger = logging.getLogger("myapp.module")


class TicTacToeTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clean_board = TicTacToeBoard(
            TicTacToeBot("p1", 1, "random"), TicTacToeBot("p2", 2, "random")
        )
        self.model = TicTacToeNet(get_input_feats(self.clean_board), cfg["nn"])
        self.optimiser = AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.rng = np.random.default_rng()  # For stochastic selection.

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
        # Training loop.
        for i, (batch_states, batch_pol_targets, batch_rewards) in enumerate(train_loader):
            # Forward pass predict policy and value.
            pred_policy, pred_value = self.model(batch_states)
            pred_policy = pred_policy.to(self.device)
            pred_value = pred_value.to(self.device)
            # Compute loss vs self play results.
            policy_loss = cross_entropy(pred_policy, batch_pol_targets)
            value_loss = mse_loss(pred_value.squeeze(), batch_rewards)
            total_loss = policy_loss + value_loss
            # Optimise.
            self.optimiser.zero_grad()  # Reset for new sample.
            total_loss.backward()
            self.optimiser.step()  # Update gradients.

        return total_loss

    def learn(self):
        for cycle in range(self.cfg["self_play"]["cycles"]):
            training_data = GameDataset()
            # Begin self play cycle.
            self.model.eval()  # Eval mode as not training yet.
            for playout in range(self.cfg["self_play"]["playouts"]):
                game = self.self_play()
                training_data.append_game(game)
            # Begin training using self play data.
            self.model.train()  # Training mode as we now will train.
            for epoch in range(self.cfg["self_play"]["epochs"]):
                epoch_loss = self.train(training_data)
                print(f"Loss for epoch {epoch}: {epoch_loss}")
            # Save model
            torch.save(self.model.state_dict(), f"models/model_{str(cycle)}.pt")
            torch.save(self.optimiser.state_dict(), f"models/optimiser_{str(cycle)}.pt")
