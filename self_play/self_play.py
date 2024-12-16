import copy
import logging

import numpy as np
import torch
from torch.optim import AdamW

from mcts.alpha_zero_mcts import SmartMCSTNode
from neural_networks.data_prep import get_input_feats
from neural_networks.tic_tac_toe_net import TicTacToeNet
from tic_tac_toe.board import TicTacToeBoard, assign_reward
from tic_tac_toe.player import TicTacToeBot

logger = logging.getLogger("myapp.module")


class TicTacToeTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.clean_board = TicTacToeBoard(
            TicTacToeBot("p1", 1, "random"), TicTacToeBot("p2", 2, "random")
        )
        self.model = TicTacToeNet(get_input_feats(self.clean_board), cfg["nn"])
        self.optimiser = AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.rng = np.random.default_rng()  # For stochastic selection.

    def self_play(self):
        # Play until termination.
        history = []  # Contains board, action probabilities and reward for each turn.
        board = copy.deepcopy(self.clean_board)
        while not board.game_result:
            # Begin search from the current position.
            mct = SmartMCSTNode(None, None, board, self.model, self.cfg["mcts"])
            actions, probs = mct.run_search()
            # Do stochastic move selection and execute.
            action = self.rng.choice(actions, p=probs)
            board.exec_move(tuple(action))
            # Update history with board state and action probabilities.
            history.append([board, probs, board.last_player])

        # Assign reward for each player based on who played for all turns.
        game_last_player = board.last_player  # Last player for game as a whole.
        for i in range(len(history)):
            # Encode board state.
            history[i][0] = get_input_feats(history[i][0])
            # Overwrite with reward.
            turn_last_player = history[i][2]
            rewards = self.cfg["mcts"]["rewards"]
            history[i][2] = assign_reward(
                game_last_player, turn_last_player, board.game_result, rewards
            )
        return history

    def train(self, mem: list):
        pass

    def learn(self):
        for cycle in range(self.cfg["self_play"]["cycles"]):
            training_data = []
            # Begin self play cycle.
            self.model.eval()  # Eval mode as not training yet.
            for playout in range(self.cfg["self_play"]["playouts"]):
                training_data += self.self_play()
            # Begin training using self play data.
            self.model.train()  # Training mode as we now will train.
            for epoch in range(self.cfg["self_play"]["epochs"]):
                self.train(training_data)
            # Save model
            torch.save(self.model.state_dict(), f"models/model_{str(cycle)}.pt")
            torch.save(self.optimiser.state_dict(), f"models/optimiser_{str(cycle)}.pt")
