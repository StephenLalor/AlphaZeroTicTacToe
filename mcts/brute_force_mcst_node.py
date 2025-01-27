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
        exploration = c_puct * np.sqrt(np.log(self.visits) / child.visits)
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
        Add a child node corresponding to new game state after an unused move has been executed.
        """
        new_board = copy.deepcopy(self.board)
        # Exec next available move for child, not random move.
        move = self.unused_moves.pop()
        new_board.exec_move(move)
        child = BruteMCSTNode(self, new_board, self.cfg)
        self.children.append(child)
        # Return the expanded child node.
        return child

    # NOTE: Changed this from checking if there are valid moves left.
    def is_fully_expanded(self):
        return len(self.children) > 0

    def traverse(self) -> "BruteMCSTNode":
        """
        Step through best child nodes if fully expanded, or expand if not.
        """
        # Start at this current node.
        node = self
        # Expand and select.
        while not node.board.game_result:
            if not node.is_fully_expanded():
                return node.expand()
            node = node.select(self.cfg["mcts"]["c"])
        return node

    def rollout(self) -> TicTacToeBoard:
        """
        From the current state, play to the end to get a result.
        """
        # Avoid modifying this node's board.
        new_board = copy.deepcopy(self.board)
        # Excute random moves until the game is over.
        while not new_board.game_result:
            rand_move = self.rng.choice(new_board.valid_moves)
            new_board.exec_move(tuple(rand_move))
        return new_board

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

    def get_reward(self, board: TicTacToeBoard) -> float:
        """
        Calculate reward for a terminated game.
        """
        player = board.last_player
        res = board.game_result
        return assign_reward(player, player, res, self.cfg["mcts"]["rewards"])

    def search(self):
        """
        Perform simulation and return the best action to take from the root node.
        """
        root = self  # Initial node itself is the root.
        for sim in range(1, self.cfg["mcts"]["sim_lim"]):
            # Do selection and expansion.
            node = root.traverse()

            # Play out game from this position.
            result_board = node.rollout()

            # Update statistics.
            value = self.get_reward(result_board)
            node.backpropagate(value)
        return root.select(0.0)
