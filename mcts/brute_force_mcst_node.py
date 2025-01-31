import copy

import numpy as np

from self_play.reward_assignment import assign_reward
from tic_tac_toe.board import TicTacToeBoard


class BruteMCSTNode:
    """
    Monte Carlo Search Tree Node.
    """

    def __init__(self, parent: "BruteMCSTNode", board: TicTacToeBoard, cfg: dict):
        self.parent = parent
        self.children = []
        self.board = board
        self.unused_moves = board.valid_moves.copy()
        self.visits = 0
        self.value = 0
        self.cfg = cfg
        self.rng = np.random.default_rng()

    def __str__(self):
        has_parent = self.parent is not None
        last_player, last_move = self.board.last_player, self.board.last_move
        struct_info = f"[Parent] {has_parent} [Children] {len(self.children)}"
        last_info = f"[Last Player] {last_player} [Last Move] {last_move}"
        uct_info = f"[Visits] {self.visits} [Value] {self.value}"
        return f"{struct_info} | {uct_info} | {last_info}"

    def calc_utc_score(self, child: "BruteMCSTNode", c_puct: float) -> float:
        """
        Calculate UTC score for child WRT parent node.
        """
        exploitation = child.value / child.visits if child.visits else 0
        exploration = c_puct * (np.sqrt(self.visits) / (child.visits + 1))
        return exploitation + exploration

    def select(self, c_puct: float) -> "BruteMCSTNode":
        """
        Choose child with highest UCT score.
        """
        best_child_node, best_score = None, -np.inf
        for child in self.children:
            score = self.calc_utc_score(child, c_puct)
            if score > best_score:
                best_score, best_child_node = score, child
        if best_child_node is None:
            raise ValueError("Failed to find best child node.")
        return best_child_node

    def expand(self) -> "BruteMCSTNode":
        """
        For each valid move, add a child node to the current node.
        """
        for move in self.board.valid_moves:
            child_board = copy.deepcopy(self.board)
            child_board.exec_move(move)
            self.children.append(BruteMCSTNode(self, child_board, self.cfg))

    def rollout(self) -> float:
        """
        From the current state, play to the end to get a result.
        """
        # Avoid modifying this node's board.
        rollout_board = copy.deepcopy(self.board)
        # Excute random moves until the game is over.
        while not rollout_board.game_result:
            rand_move = self.rng.choice(rollout_board.valid_moves)
            rollout_board.exec_move(tuple(rand_move))
        # Determine reward for terminal game.
        player = rollout_board.last_player
        return assign_reward(player, player, rollout_board.game_result, self.cfg["mcts"]["rewards"])

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
            # Negate value as player has switched.
            value = -value

    def is_fully_expanded(self):
        return len(self.children) > 0

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

    def search(self):
        root = self
        for sim in range(1, self.cfg["mcts"]["sim_lim"] + 1):
            # Begin every simulation from root node.
            node = root
            # Selection phase.
            while node.is_fully_expanded() and not node.board.game_result:
                node = node.select(self.cfg["mcts"]["c"])
            # Expansion phase.
            if not node.is_fully_expanded() and not node.board.game_result:
                node.expand()
            # Simulation phase.
            value = node.rollout()
            # Update phase.
            node.backpropagate(value)

        return root.get_action_prob_dist()
