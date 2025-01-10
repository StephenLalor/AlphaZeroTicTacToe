import copy
import logging
import uuid
from collections import deque

import numpy as np
import torch

from neural_networks.data_prep import get_input_feats, policy_to_valid_moves
from neural_networks.tic_tac_toe_net import TicTacToeNet
from tic_tac_toe.board import TicTacToeBoard, assign_reward

logger = logging.getLogger("myapp.module")


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
        cfg: dict,
    ):
        self.uuid = uuid.uuid4()
        self.parent = parent
        self.children = []
        self.action_probs = None
        self.board = board
        self.visits = 0
        self.value = 0
        self.score = 0  # Selection score.
        self.prior = prior_prob
        self.cfg = cfg
        self.model = model

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

    def calc_selection_score(self, child: "SmartMCSTNode", c_puct: float) -> float:
        """
        Calculate score using PUCT and mean value for child WRT parent node.

        Initially larger for moves with high prior probability, since visits will be small. Over
        time value increases and that term then dominates.
        """
        q_value = child.value / child.visits if child.visits else 0
        puct = c_puct * child.prior * (np.sqrt(self.visits) / (child.visits + 1))
        logger.debug(f"Q: {q_value}, PUCT: {puct}")
        return q_value + puct

    def select(self, c_puct: float) -> "SmartMCSTNode":
        """
        Choose child with highest selection score.
        """
        best_child_node, best_score = None, -np.inf
        for child in self.children:
            score = self.calc_selection_score(child, c_puct)
            child.score = score
            if score > best_score:
                logger.debug(f"New best selection score {child.uuid} \n\t{child}.")
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
        logger.debug(f"Adding child nodes for policy \n\t{policy}.")
        for move, prob in policy.items():
            new_board = copy.deepcopy(self.board)
            new_board.exec_move(move)
            self.children.append(SmartMCSTNode(self, prob, new_board, self.model, self.cfg))

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
        player = self.board.last_player  # Player who moved last.
        res = self.board.game_result
        return assign_reward(player, player, res, self.cfg["rewards"])

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
    def search(self):
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
        root = self
        logger.debug(f"Beggining search. \n\t{root}.")
        for sim in range(1, self.cfg["sim_lim"] + 1):
            # Begin each simulation at the root node.
            logger.debug(f"Search iteration {sim}.")
            node = root

            # Selection phase.
            while node.is_fully_expanded():  # Skipped on first simulation.
                logger.debug(f"Fully expanded, doing selection on {node.uuid} \n\t{node}.")
                node = node.select(self.cfg["c"])
                logger.debug(f"Selected node {node.uuid} \n\t{node}.")

            # Expansion phase.
            if not node.is_fully_expanded():
                logger.debug(f"Not fully expanded {node.uuid} \n\t{node}")
                if not node.board.game_result:
                    logger.debug("Game state not terminal. Doing evaluation with NN.")
                    # Evaluation - use predicted reward from value head.
                    input_feats = get_input_feats(node.board)
                    policy, value = self.model(input_feats)
                    policy, value = self.model.parse_output(policy, value)
                    logger.debug(f"\tPolicy: {policy} Value: {value}.")
                    # Expansion.
                    logger.debug(f"Expanding {node.uuid} \n\t{node}")
                    node.expand(policy)
                else:
                    # Result known so no need for value head, use defined reward instead.
                    logger.debug("Game state is terminal. Doing evaluation with defined rewards.")
                    policy, value = None, node.get_reward()
                    logger.debug(f"\tPolicy: {policy} Value: {value}.")

            # Update phase.
            logger.debug(f"Updating from {node.uuid} \n\t{node}")
            node.backpropagate(value)
            # Yield so we can request next step from front end.
            yield root

        # Get probability distribution of all actions available from root.
        self.action_probs = root.get_action_prob_dist()

    def run_search(self) -> tuple[np.array, np.array]:
        """
        Utility to run the generator to exhaustion.
        """
        # Yield but do not store by using zero max length.
        deque(self.search(), maxlen=0)  # Efficiently exhausts generator.
        return self.action_probs
