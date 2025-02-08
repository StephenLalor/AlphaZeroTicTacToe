import copy
import logging
import math
import uuid

import numpy as np
import torch

from neural_networks.data_prep import get_input_feats, policy_to_valid_moves
from neural_networks.tic_tac_toe_net import TicTacToeNet
from self_play.reward_assignment import assign_reward
from tic_tac_toe.board import TicTacToeBoard

logger = logging.getLogger("myapp.module")


class MCSTNode:
    """
    Monte Carlo Search Tree Node.

    Rollouts are replaced by the value from our neural network.
    """

    def __init__(
        self,
        parent: "MCSTNode",
        prior_prob: float,
        board: TicTacToeBoard,
        cfg: dict,
    ):
        self.uuid = uuid.uuid4()
        self.parent = parent
        self.children = []
        self.board = board
        self.visits = 0
        self.value = 0
        self.score = 0  # Selection score.
        self.prior = prior_prob
        self.cfg = cfg

    def __str__(self):
        has_parent = self.parent is not None
        last_player, last_move = self.board.last_player, self.board.last_move
        struct_info = f"[Parent]: {has_parent} [Children]: {len(self.children)}"
        last_info = f"[Last Player]: {last_player.name} [Last Move]: {last_move}"
        uct_info = f"[Visits]: {self.visits} [Value]: {self.value}"
        prior_info = f"[Prior]: {np.round(self.prior, 3) if self.prior is not None else 0}"
        score_info = f"[Score]: {np.round(self.score, 3) if self.score is not None else 0}"
        return f"{struct_info} | {uct_info} | {last_info} | {prior_info} | {score_info}"

    def is_fully_expanded(self):
        return len(self.children) > 0

    def calc_selection_score(self, child: "MCSTNode", c_puct: float) -> float:
        """
        Calculate score using PUCT and mean value for child WRT parent node.

        Initially larger for moves with high prior probability, since visits will be small. Over
        time Q-value increases and that term then dominates.

        The parent wants the child to have a low Q-value as it wants the opponent to be in a bad
        situation, so we invert the Q-value by subtracting it from 1. Low Q-value for child means
        high Q-value for parent.
        """
        # Invert Q-value and scale [-1, 1] → [0, 1].
        q_value = 1 - ((child.value / child.visits + 1) / 2) if child.visits else 0.0
        # Calculate puct score using prior selection probability.
        puct = c_puct * child.prior * math.sqrt(self.visits) / (child.visits + 1)
        return q_value + puct

    def select(self, c_puct: float) -> "MCSTNode":
        """
        Choose child with highest selection score.
        """
        best_child_node, best_score = None, -np.inf
        for child in self.children:
            child.score = self.calc_selection_score(child, c_puct)
            if child.score > best_score:
                best_score, best_child_node = child.score, child
        if best_child_node is None:
            raise ValueError("Failed to find best child node.")
        return best_child_node

    def expand(self, policy: torch.Tensor):
        """
        For each move determined by the policy, add a child to the current node.
        """
        # Add a child node for every valid move.
        for move, prob in policy.items():
            if prob > 0:
                new_board = copy.deepcopy(self.board)
                new_board.exec_move(move)
                self.children.append(MCSTNode(self, prob, new_board, self.cfg))

    def backpropagate(self, value: float):
        """
        From current node back up to the root node, update stats.

        If the child node wins, then this node must have lost. Hence we negate the reward.
        """
        node = self  # Starting at the current node.
        while node is not None:
            # Update metrics for the current node.
            node.visits += 1
            node.value += value
            # Traverse up to parent.
            node = node.parent
            value = -value  # Negate value as player has switched.

    def get_reward(self) -> float:
        """
        Calculate reward for a terminated game.
        """
        player = self.board.last_player
        res = self.board.game_result
        return assign_reward(player, player, res, self.cfg["mcts"]["rewards"])

    def get_action_prob_dist(self) -> tuple[np.array, np.array]:
        """
        Get normalised action probability based on number of visits for each child node.
        """
        tot_visits = sum(child.visits for child in self.children)
        probs = np.zeros(len(self.board.moves), dtype=np.float32)
        actions_map = {action: i for i, action in enumerate(self.board.moves)}
        for child in self.children:
            if child.board.last_move in actions_map:
                probs[actions_map[child.board.last_move]] = child.visits / tot_visits
        actions = np.array(self.board.moves)
        return (actions, probs)


@torch.no_grad()
def search(board: TicTacToeBoard, model: TicTacToeNet, cfg: dict) -> tuple[np.array, np.array]:
    """
    Selection:
        - Starting from the root, recursively select child nodes using the UCT formula.
        - This continues until you reach a node that is not fully expanded (a leaf node).
            - This node might be many levels deep in the tree.

    Expansion:
        - At this leaf node, perform expansion, creating its children.

    Evaluation:
        - Evaluate the newly created child nodes (or the leaf node if it's a terminal node).
        - Done using Policy and Value networks.

    Backpropagation:
        - The results of the evaluation are backpropagated up the tree
        - From the expanded leaf node back to the root.
        - Nodes along this path have visit counts and values updated.
    """
    # Begin each simulation at a new root node.
    root = MCSTNode(None, 0, board, cfg)
    root.visits = 1
    # Apply dirchlet noise to first expansion.
    feats = get_input_feats(root.board)
    policy, _ = model(feats.to(model.device))
    policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
    eps, alpha = cfg["mcts"]["dirichlet_epsilon"], cfg["mcts"]["dirichlet_alpha"]
    noisy_policy = (1 - eps) * policy + eps * np.random.dirichlet([alpha] * 9)
    noisy_policy = policy_to_valid_moves(noisy_policy, root.board.moves, root.board.valid_moves)
    root.expand(noisy_policy)

    # Begin search from expanded root node.
    for sim in range(1, cfg["mcts"]["sim_lim"] + 1):
        # Selection phase.
        node = root
        while node.is_fully_expanded():  # Skipped on first simulation.
            node = node.select(cfg["mcts"]["c"])

        # Expansion phase.
        if not node.board.game_result:
            # Evaluation - use predicted reward from value head.
            feats = get_input_feats(node.board)
            policy, value = model(feats.to(model.device))
            policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
            policy = policy_to_valid_moves(policy, node.board.moves, node.board.valid_moves)
            value = value.item()  # No negation!
            # Expansion.
            node.expand(policy)
        # Result known so no need for value head, use defined reward instead.
        else:
            # Last action taken was by parent node, because expand happens before this check.
            policy, value = None, -node.get_reward()

        # Update phase.
        node.backpropagate(value)

    # Get probability distribution of all actions available from root.
    return root.get_action_prob_dist()
