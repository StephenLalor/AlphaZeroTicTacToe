import copy
import logging

import numpy as np
import torch

from mcts.alpha_zero_mcts import SmartMCSTNode
from neural_networks.data_prep import get_input_feats
from neural_networks.tic_tac_toe_net import TicTacToeNet
from tic_tac_toe.board import TicTacToeBoard

logger = logging.getLogger("myapp.module")


class TicTacToeTrainer:
    def __init__(
        self,
        model: TicTacToeNet,
        optimiser: torch.optim.Optimizer,
        board: TicTacToeBoard,
        cfg: dict,
    ):
        self.cfg = cfg
        self.board = board
        self.optimiser = optimiser
        self.model = TicTacToeNet(get_input_feats(board), cfg["hparams"])
        self.mct = SmartMCSTNode(None, None, board, model, cfg["opts"])

    def self_play(self):
        # Set up for this play instance.
        history = []
        board = copy.deepcopy(self.board)  # Board will be edited during play.
        # Play until termination.
        while not board.game_result:
            # Generate moves and their probabilities from MC search.
            actions = self.mct.run_search()
            # Do stochastic move selection and execute.
            action = np.random.choice(actions.keys(), p=actions.values())
            board.exec_move(action)
            # Update history.
            history.append([get_input_feats(board), actions, None])
        # Assign reward based on game outcome to all.
        # TODO: Refactor this, it is crappy. Board can just handle this logic itself.
        if board.game_result == "draw":
            reward = self.cfg["opts"]["draw"]
        if board.last_player == board.game_result:
            reward = self.cfg["opts"]["win"]
        if board.last_player != board.game_result:
            reward = self.cfg["opts"]["lose"]
        # Track data.
        for i in range(len(history)):
            history[i][2] = reward
        return history

    def train(self, mem: list):
        pass

    def learn(self):
        for cycle in range(self.opts["train_cycle_lim"]):
            training_data = []  # TODO: Rename this.
            # Begin self play cycle.
            self.model.eval()  # Eval mode as not training yet.
            for playout in range(self.opts["self_play_lim"]):
                training_data += self.self_play()
            # Begin training using self play data.
            self.model.train()  # Training mode as we now will train.
            for epoch in range(self.opts["epoch_lim"]):
                self.train(training_data)
            # Save model
            torch.save(self.model.state_dict(), f"models/{str(cycle)}/model.pt")
            torch.save(self.optimiser.state_dict(), f"models/{str(cycle)}/optimiser.pt")
