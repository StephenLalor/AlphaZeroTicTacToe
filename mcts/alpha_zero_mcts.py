import copy

import numpy as np
import torch

from neural_networks.data_prep import get_input_feats, policy_to_valid_moves
from neural_networks.tic_tac_toe_net import TicTacToeNet
from tic_tac_toe.board import TicTacToeBoard, TicTacToeException


class SmartMCSTNode:
    """
    Monte Carlo Search Tree Node.

    Rollouts are replaced by the value from our neural network.
    """

    def __init__(
        self,
        parent: "SmartMCSTNode",
        prior_prob: float,
        board: TicTacToeBoard,
        model: TicTacToeNet,
        opts: dict,
    ):
        self.parent = parent
        self.children = []
        self.opt_child = None
        self.board = board
        self.unused_moves = board.valid_moves.copy()
        self.visits = 0
        self.value = 0
        self.prior_prob = prior_prob
        self.opts = opts
        self.model = model

    def __str__(self):
        has_parent = self.parent is not None
        last_player, last_move = self.board.last_player, self.board.last_move
        struct_info = f"[Parent]: {has_parent} [Children]: {len(self.children)}"
        last_info = f"[Last Player]: {last_player.name} [Last Move]: {last_move}"
        uct_info = f"[Visits]: {self.visits} [Value]: {self.value}"
        prior_info = f"[Prior]: {self.prior_prob}"
        return f"{struct_info} | {uct_info} | {last_info} | {prior_info}"

    def is_fully_expanded(self):
        return len(self.children) > 0

    def calc_selection_score(self, child: "SmartMCSTNode", c_puct: float) -> float:
        """
        Calculate score using PUCT and mean value for child WRT parent node.

        Initially larger for moves with high prior probability, since visits will be small. Over
        time value increases and that term then dominates.
        """
        # TODO: Consider mapping to values in [-1, 1].
        q_value = child.value / child.visits if child.visits else 0
        puct = c_puct * child.prior_prob * (np.sqrt(self.visits) / (child.visits + 1))
        return q_value + puct

    def select(self, c_puct: float) -> "SmartMCSTNode":
        """
        Choose child with highest selection score.
        """
        best_child_node, best_score = None, -np.inf
        for child in self.children:
            score = self.calc_selection_score(child, c_puct)
            if score > best_score:
                best_score, best_child_node = score, child
        if best_child_node is None:
            raise ValueError("Failed to find best child node.")
        return best_child_node

    def expand(self, policy: torch.Tensor):
        """
        For each move determined by the policy, add a child to the current node.
        """
        # Add a child node for every valid move.
        policy = policy_to_valid_moves(policy, self.board.moves, self.board.valid_moves)
        for move, prob in policy.items():
            new_board = copy.deepcopy(self.board)
            new_board.exec_move(move)
            self.children.append(SmartMCSTNode(self, prob, new_board, self.model, self.opts))

    def backpropagate(self, value: float):
        """
        From current node back up to the root node, update stats.
        """
        node = copy.deepcopy(self)  # Starting at the current node.
        while node is not None:
            node.visits += 1
            # Avoid updating score if we're on the root node.
            if node.parent:
                node.value += value
            # Parent node is alternate player, so negate value.
            value = -value
            node = node.parent

    def get_reward(self) -> float:
        """
        Calculate reward for a terminated game.
        """
        if self.board.game_result == "draw":
            return self.opts["draw"]
        if self.board.last_player == self.board.game_result:
            return self.opts["win"]
        if self.board.last_player != self.board.game_result:
            return self.opts["lose"]
        # Error if result not as expected.
        allowed = [self.board.p1, self.board.p2, "draw"]
        err_msg = f"Game result {self.board.game_result} not one of {allowed}"
        raise TicTacToeException(err_msg)

    @torch.no_grad()
    def search(self):
        root = node = self
        for sim in range(1, self.opts["sim_lim"] + 1):  # TODO: Add a time limit also.
            # Selection phase.
            while node.is_fully_expanded():
                node = node.select(self.opts["c"])

            # Expansion phase.
            if not node.board.game_result:
                # Use predicted reward from value head.
                input_feats = get_input_feats(node.board)
                policy, value = self.model(input_feats)
                policy, value = self.model.parse_output(policy, value)
                node.expand(policy)
            else:
                # Result known so no need for value head, use defined reward instead.
                policy, value = None, node.get_reward()

            # Update phase.
            node.backpropagate(value)
            yield root  # So we can visualise tree on each update.

        # Finally find the best child node with respect to the root.
        self.opt_child = root.select(0.0)
