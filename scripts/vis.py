import matplotlib.pyplot as plt
import mlflow

from neural_networks.data_prep import get_input_feats
from neural_networks.load_model import load_mdl_for_inference
from tic_tac_toe.board import TicTacToeBoard
from tic_tac_toe.player import TicTacToeBot

mlflow.set_tracking_uri("http://localhost:5000")
mdl = load_mdl_for_inference("test_simple", 2)

# Hook function to extract activations.
activations = {}


def get_activation(name):
    def hook(mdl, nn_in, nn_out):
        activations[name] = nn_out.detach()

    return hook


# Register hooks to layers.
mdl.conv_block.conv.register_forward_hook(get_activation("conv_block.conv"))
mdl.conv_block.conv.register_forward_hook(get_activation("conv_block.relu"))

# Forward pass on sample input.
board = TicTacToeBoard(TicTacToeBot("p1", "X"), TicTacToeBot("p2", "O"))
board.exec_move((0, 0))  # X
board.exec_move((2, 2))  # O
board.exec_move((0, 1))  # X
board.exec_move((1, 1))  # O
# board.exec_move((0, 2))  # X <-- X wins
feats = get_input_feats(board)
pol, val = mdl(feats)
pol, val = mdl.parse_output(pol, val)
pos = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
pol = [(y, round(float(x), 2)) for x, y in zip(pol, pos)]
print(board.next_player.name)
print(board)
print(pol)
print(val)

board = TicTacToeBoard(TicTacToeBot("p1", "X"), TicTacToeBot("p2", "O"))
board.exec_move((2, 0))  # X
board.exec_move((1, 0))  # O
board.exec_move((0, 1))  # X
board.exec_move((1, 1))  # O
board.exec_move((2, 0))  # X
# board.exec_move((1, 2))  # O  <-- O wins
feats = get_input_feats(board)
pol, val = mdl(feats)
pol, val = mdl.parse_output(pol, val)
pos = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
pol = [(y, round(float(x), 2)) for x, y in zip(pol, pos)]
print(board.next_player.name)
print(board)
print(pol)
print(val)

# NOTE: Very bad val prediction for diagonal wins?
board = TicTacToeBoard(TicTacToeBot("p1", "X"), TicTacToeBot("p2", "O"))
board.exec_move((0, 0))  # X
board.exec_move((0, 1))  # O
board.exec_move((1, 1))  # X
board.exec_move((0, 2))  # O
# board.exec_move((2, 2))  # X  <-- X wins
feats = get_input_feats(board)
pol, val = mdl(feats)
pol, val = mdl.parse_output(pol, val)
pos = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
pol = [(y, round(float(x), 2)) for x, y in zip(pol, pos)]
print(board.next_player.name)
print(board)
print(pol)
print(val)

# NOTE: Very bad val prediction for diagonal wins?
board = TicTacToeBoard(TicTacToeBot("p1", "X"), TicTacToeBot("p2", "O"))
board.exec_move((0, 0))  # X
board.exec_move((0, 2))  # O
board.exec_move((0, 1))  # X
board.exec_move((1, 1))  # O
board.exec_move((2, 1))  # X
# board.exec_move((2, 0))  # O  <-- O wins
feats = get_input_feats(board)
pol, val = mdl(feats)
pol, val = mdl.parse_output(pol, val)
pos = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
pol = [(y, round(float(x), 2)) for x, y in zip(pol, pos)]
print(board.next_player.name)
print(board)
print(pol)
print(val)


# Check
print(activations.keys())  # Check which layers were hooked
print(activations["conv_block.relu"].shape)

# Visualise.
print(board.state)

data = {}
plt.figure(figsize=(15, 15))
for i in range(64):
    # Softmax activations over whole tensor.
    # filter_activations = activations["conv_block.relu"][0, i, :, :]
    # filter_activations = filter_activations.view(-1)  # Flatten.
    # filter_activations = nn.functional.softmax(filter_activations, dim=0)
    # plt_data = filter_activations.view(3, 3)  # Back to grid.
    # data[i] = plt_data
    plt_data = activations["conv_block.relu"][0, i, :, :]
    data[i] = plt_data
    plt.subplot(8, 8, i + 1)
    plt.imshow(plt_data, cmap="gray")
    plt.title(f"Filter {i}")
    plt.axis("off")
    plt.colorbar()
plt.show()
