from tic_tac_toe.board import TicTacToeBoard
from tic_tac_toe.mcts import MCSTNode
from tic_tac_toe.parse_tree import parse_mcst
from tic_tac_toe.player import TicTacToeBot

opts = {"sim_lim": 18, "c": 1.4, "win": 1.0, "lose": -1.0, "draw": 0.5}
p1 = TicTacToeBot("random_p1", "X", "random")
p2 = TicTacToeBot("random_p2", "O", "random")
board = TicTacToeBoard(p1, p2)
root_node = MCSTNode(None, board, opts)
all(root_node.sim())

# Investigate.

print(root_node)
for child in root_node.children:
    print(child)

# Parse.
parsed_tree = parse_mcst(root_node)
parsed_tree
