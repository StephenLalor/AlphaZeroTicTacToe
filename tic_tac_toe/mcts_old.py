import copy

import numpy as np
from loguru import logger

from tic_tac_toe.board import TicTacToeException

# Set up logging file.
logger.add("mcts.log", mode="w")


class MonteCarloTreeNode:
    def __init__(self, parent_node, state, last_player, opts):
        self.parent_node = parent_node
        self.child_nodes = []
        self.visit_count = 0
        self.value_sum = 0
        self.state = state
        self.opts = opts
        self.last_player = last_player

    def __str__(self):
        num_child = len(self.child_nodes)
        has_parent = self.parent_node is not None
        player = self.last_player.name if self.last_player else None
        return f"Visits: {self.visit_count}, Value: {self.value_sum}, Children: {num_child}, Parent: {has_parent}, Last Player: {player}"

    def is_fully_expanded(self):
        return len(self.state.valid_moves) == 0

    def calc_utc_score(self, child_node):
        exploitation_term = child_node.value_sum / child_node.visit_count
        exploration_term = np.sqrt(np.log(self.visit_count) / child_node.visit_count)
        return exploitation_term + (self.opts["c"] * exploration_term)

    def get_best_child_node(self):
        best_child_node, best_score = None, -np.inf
        for child_node in self.child_nodes:
            score = self.calc_utc_score(child_node)
            if score > best_score:
                best_score, best_child_node = score, child_node
        if best_child_node is None:
            raise ValueError("Failed to find best child node.")
        return best_child_node


class MonteCarloTree:
    def __init__(self, root_node, opts, player_1, player_2):
        self.root_node = root_node
        self.opts = opts
        self.player_1 = player_1
        self.player_2 = player_2

    def expand_node(self, node):
        # Generate new board state with the next player's move.
        new_board = copy.deepcopy(node.state)
        curr_player = (
            self.player_1
            if (node.last_player in [None, self.player_2])
            else self.player_2
        )
        move = curr_player.generate_move(new_board.valid_moves)
        new_board.exec_move(move)
        # Add new child node with this board state and update last player.
        child_node = MonteCarloTreeNode(node, new_board, curr_player, self.opts)
        node.child_nodes.append(child_node)
        return child_node

    def rollout(self, node):
        # Set initial player and prepare for rollout from this state.
        logger.debug(f"Node: {node}")
        curr_state = copy.deepcopy(
            node.state
        )  # We just want result of playing from this state.
        curr_player = (
            self.player_1
            if (node.last_player in [None, self.player_2])
            else self.player_2
        )
        logger.debug(f"Initial board state:\n {curr_state}")

        # Return result early if terminal node.
        if curr_state.game_result:
            return curr_player if curr_state.game_result == "win" else "draw"

        # Play out until terminal state.
        while not curr_state.game_result:
            # Generate and execute the current player's move.
            curr_state.exec_move(curr_player.generate_move(curr_state.valid_moves))
            logger.debug(f"New board state:\n {curr_state}")
            # Check for game result after the move.
            if curr_state.game_result:
                return curr_player if curr_state.game_result == "win" else "draw"
            # Alternate players.
            curr_player = (
                self.player_1 if curr_player == self.player_2 else self.player_2
            )
        raise TicTacToeException("Game did not terminate with win or draw")

    def backpropagate(self, node, result):
        curr_node = node
        while curr_node is not None:
            # Determine the reward based on the outcome.
            if result == "draw":
                reward = self.opts["draw"]
            if curr_node.last_player == result:
                reward = self.opts["win"]
            if curr_node.last_player != result:
                reward = self.opts["lose"]

            # Update node stats and move back up to parent node.
            curr_node.visit_count += 1
            if curr_node.parent_node:  # Avoid updating if we're on the root node.
                curr_node.value_sum += reward
            curr_node = curr_node.parent_node

    def traverse(self, node):
        while not node.state.game_result:
            if not node.is_fully_expanded():
                return self.expand_node(node)
            node = node.get_best_child_node()
        return node

    def sim(self):
        curr_node = self.root_node  # Start at root node.
        for sim in range(self.opts["sim_lim"]):
            logger.info(f"Simulating ({sim}/{self.opts["sim_lim"]})")
            curr_node = self.traverse(curr_node)
            res = self.rollout(curr_node)
            self.backpropagate(curr_node, res)
