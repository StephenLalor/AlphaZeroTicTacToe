from tic_tac_toe.board import TicTacToeBoard
from tic_tac_toe.mcts import MCTNode
from tic_tac_toe.player import TicTacToeBot

opts = {"sim_lim": 5, "c": 1.4, "win": 1.0, "lose": -1.0, "draw": 0.5}
p1 = TicTacToeBot("random_p1", "X", "random")
p2 = TicTacToeBot("random_p2", "O", "random")
board = TicTacToeBoard(p1, p2)
root_node = MCTNode(None, board, opts)
best_move = root_node.sim()

# Investigate.
print(root_node)
print(best_move)
for child in root_node.children:
    print(child.board.last_move)
