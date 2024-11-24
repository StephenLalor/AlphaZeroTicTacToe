from mcts.alpha_zero_mcts import SmartMCSTNode
from neural_networks.data_prep import get_input_feats
from neural_networks.tic_tac_toe_net import TicTacToeNet
from tic_tac_toe.board import TicTacToeBoard
from tic_tac_toe.player import TicTacToeBot

hparams = {"hidden": 64, "res_blocks": 4, "pol_feats": 32, "val_feats": 3}
opts = {"sim_lim": 9, "c": 1.4, "win": 1.0, "lose": -1.0, "draw": 0.5}
p1 = TicTacToeBot("random_p1", 1, "random")
p2 = TicTacToeBot("random_p2", 2, "random")
board = TicTacToeBoard(p1, p2)
dummy_input = get_input_feats(board)
model = TicTacToeNet(dummy_input, hparams)

root_node = SmartMCSTNode(parent=None, prior_prob=None, board=board, model=model, opts=opts)
for step in root_node.search():
    print(step)

opt_child = root_node.opt_child
print(opt_child)
