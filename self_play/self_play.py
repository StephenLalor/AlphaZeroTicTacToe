import copy
import logging

import numpy as np
import torch
from torch.nn.functional import cross_entropy, mse_loss
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from mcts.alpha_zero_mcts import SmartMCSTNode
from neural_networks.data_prep import get_input_feats
from neural_networks.tic_tac_toe_net import TicTacToeNet
from tic_tac_toe.board import TicTacToeBoard, assign_reward
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

    def self_play(self) -> TensorDataset:
        # Board, action probabilities and reward for each turn.
        board_hist, player_hist, probs_hist, reward_hist = [], [], [], []
        # Play until termination.
        board = copy.deepcopy(self.clean_board)
        while not board.game_result:
            # Begin search from the current position.
            mct = SmartMCSTNode(None, None, board, self.model, self.cfg["mcts"])
            actions, probs = mct.run_search()
            # Do stochastic move selection and execute.
            action = self.rng.choice(actions, p=probs)
            board.exec_move(tuple(action))
            # Update history with tensors.
            board_hist.append(get_input_feats(board).squeeze(0))  # Batch dim re-added later.
            player_hist.append(board.last_player)
            probs_hist.append(torch.from_numpy(probs))

        # Assign reward for each turn now that result is known.
        res = board.game_result
        last_player = board.last_player  # Last player for game as a whole.
        for turn_player in player_hist:
            turn_reward = assign_reward(last_player, turn_player, res, self.cfg["mcts"]["rewards"])
            reward_hist.append(torch.tensor(turn_reward, dtype=torch.float32))

        # Collect history as tensor stacks.
        return torch.stack(board_hist), torch.stack(probs_hist), torch.stack(reward_hist)

    def train(self, training_data: list):
        # Set up batch loader with shuffling.
        batch_size = self.cfg["nn"]["batch_size"]
        states = torch.cat([data[0] for data in training_data])
        probs = torch.cat([data[1] for data in training_data])
        rewards = torch.cat([data[2] for data in training_data])
        history_ds = TensorDataset(states, probs, rewards)
        train_loader = DataLoader(history_ds, batch_size=batch_size, shuffle=True)
        # Training loop.
        for i, (states, probs, rewards) in enumerate(train_loader):
            # Forward pass predict policy and value.
            states = states.to(self.device)
            pred_policy, pred_value = self.model(states)  # TODO: Move to device.
            pred_policy = pred_policy.to(self.device)
            pred_value = pred_value.to(self.device)
            # Compute loss vs self play results.
            probs = probs.to(self.device)
            policy_loss = cross_entropy(pred_policy, probs)
            rewards = rewards.to(self.device)
            value_loss = mse_loss(pred_value.squeeze(), rewards)
            total_loss = policy_loss + value_loss
            # Optimise.
            self.optimiser.zero_grad()  # Reset for new sample.
            total_loss.backward()
            self.optimiser.step()  # Update gradients.
        return total_loss

    def learn(self):
        for cycle in range(self.cfg["self_play"]["cycles"]):
            training_data = []
            # Begin self play cycle.
            self.model.eval()  # Eval mode as not training yet.
            for playout in range(self.cfg["self_play"]["playouts"]):
                playout_history = self.self_play()
                training_data.append(playout_history)
            # Begin training using self play data.
            self.model.train()  # Training mode as we now will train.
            for epoch in range(self.cfg["self_play"]["epochs"]):
                epoch_loss = self.train(training_data)
                print(f"Loss for epoch {epoch}: {epoch_loss}")
            # Save model
            torch.save(self.model.state_dict(), f"models/model_{str(cycle)}.pt")
            torch.save(self.optimiser.state_dict(), f"models/optimiser_{str(cycle)}.pt")
