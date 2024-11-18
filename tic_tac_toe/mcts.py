import copy

import numpy as np
from loguru import logger

from tic_tac_toe.board import TicTacToeBoard, TicTacToeException
from tic_tac_toe.player import TicTacToeBot

# Set up logging file.
logger.add("tic_tac_toe/mcts.log", mode="w")


class MCSTNode:
    """
    Monte Carlo Search Tree Node.
    """

    def __init__(self, parent: "MCSTNode", board: TicTacToeBoard, opts: dict):
        self.parent = parent
        self.children = []
        self.board = board
        self.unused_moves = board.valid_moves
        self.visits = 0
        self.value = 0
        self.opts = opts

    def __str__(self):
        has_parent = self.parent is not None
        last_player, last_move = self.board.last_player, self.board.last_move
        struct_info = f"[Parent] {has_parent} [Children] {len(self.children)}"
        last_info = f"[Last Player] {last_player} [Last Move] {last_move}"
        uct_info = f"[Visits] {self.visits} [Value] {self.value}"
        return f"{struct_info} | {uct_info} | {last_info}"

    def calc_utc_score(self, child: "MCSTNode", c: float) -> float:
        """
        Calculate UTC score for child WRT parent node.
        """
        exploitation = child.value / child.visits
        exploration = np.sqrt(np.log(self.visits) / child.visits)
        return exploitation + (c * exploration)

    def get_best_child(self, c: float) -> "MCSTNode":
        """
        Choose child with highest UCT score.
        """
        best_child_node, best_score = None, -np.inf
        for child in self.children:
            score = self.calc_utc_score(child, c)
            if score > best_score:
                best_score, best_child_node = score, child
        if best_child_node is None:
            raise ValueError("Failed to find best child node.")
        return best_child_node

    def expand(self) -> "MCSTNode":
        """
        Add a child node corresponding to new game state after an unused move has been executed.
        """
        new_board = copy.deepcopy(self.board)
        # Exec next available move for child, not random move.
        new_board.exec_move(self.unused_moves.pop())
        child = MCSTNode(self, new_board, self.opts)
        self.children.append(child)
        return child  # Return the expanded child node.

    def is_fully_expanded(self) -> bool:
        """
        Node is fully expanded if there are no moves remaining that have not already been tried.
        """
        return len(self.unused_moves) == 0

    def traverse(self) -> "MCSTNode":
        """
        Step through best child nodes if fully expanded, or expand if not.
        """
        node = self  # Start at this current node.
        while not node.board.game_result:
            if not node.is_fully_expanded():
                return node.expand()
            node = node.get_best_child(self.opts["c"])
        return node

    def rollout(self) -> TicTacToeBot | str:
        """
        From the current state, play to the end to get a result.
        """
        logger.debug(f"Node: {self}")
        board = copy.deepcopy(self.board)
        logger.debug(f"Initial board state:\n{board}")
        while not board.game_result:
            # Directly use player's move generation (now random move).
            player = board.get_next_player()
            move = player.generate_move(board.valid_moves)
            board.exec_move(move[0])
            logger.debug(f"New board state:\n{board}")
            if board.game_result:
                return player if board.game_result == "win" else "draw"
        return board.game_result

    def get_reward(self, node: "MCSTNode", game_result: TicTacToeBot | str) -> float:
        """
        Calculate reward for a terminated game.
        """
        if game_result == "draw":
            return self.opts["draw"]
        if node.board.last_player == game_result:
            return self.opts["win"]
        if node.board.last_player != game_result:
            return self.opts["lose"]
        err_msg = f"Game result {game_result} not one of {[node.board.p1, node.board.p2, "draw"]}"
        raise TicTacToeException(err_msg)

    def backpropagate(self, game_result: TicTacToeBot | str):
        """
        From current node back up to the root node, update stats.
        """
        node = self  # Starting at the current node.
        while node is not None:
            node.visits += 1
            if node.parent:  # Avoid updating score if we're on the root node.
                node.value += self.get_reward(node, game_result)
            node = node.parent

    def sim(self):
        """
        Perform simulation and return the best action to take from the root node.
        """
        root = self  # Initial node itself is the root.
        for sim in range(1, self.opts["sim_lim"]):
            logger.info(f"Simulating ({sim}/{self.opts["sim_lim"]})")
            node = root.traverse()
            game_result = node.rollout()
            node.backpropagate(game_result)
        best_child = root.get_best_child(0.0)
        logger.info(best_child)
        return best_child.board.last_move
