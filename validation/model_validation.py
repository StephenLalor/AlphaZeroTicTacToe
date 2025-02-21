import math

import matplotlib.pyplot as plt
import numpy as np


def plot_state_assessment(board, policy, value, policy_act, value_act):
    _, axes = plt.subplots(2, 2, figsize=(16, 7))
    plot_board_state(axes[0], board)
    plot_move_probs(axes[1], board, policy, value)
    plt.tight_layout()
    plt.show()


def plot_board_state(board_ax, board):
    """
    Plot TicTacToe board state by placing symbols on grid.
    """
    for row in range(board.dim[0]):
        for col in range(board.dim[1]):
            symbol = ""
            if board.state[row][col] == board.p1.val:
                symbol = board.p1.symbol
            elif board.state[row][col] == board.p2.val:
                symbol = board.p2.symbol
            board_ax.text(col, row, symbol, ha="center", va="center", fontsize=60, color="black")
            board_ax.text(col - 0.4, row + 0.4, f"{(row, col)}", fontsize=20, color="gray")
    board_ax.set_title("Board State")
    board_ax.set_xticks(np.arange(-0.5, 3.0, 1))
    board_ax.set_yticks(np.arange(-0.5, 3.0, 1))
    board_ax.grid(which="major", color="black", linestyle="-", linewidth=2)
    board_ax.tick_params(axis="both", which="both", length=0)
    board_ax.set_xticklabels([])
    board_ax.set_yticklabels([])
    board_ax.set_aspect("equal")  # Ensure board remains square.
    board_ax.invert_yaxis()  # Shift (0,0) to be top left instead of bottom left.


def plot_move_probs(probs_ax, board, policy, value):
    """
    Plot bar chart of NN move probability distribution.
    """
    bar_colors = ["blue"] * len(board.moves)
    bar_colors[np.argmax(policy)] = "green"  # Highlight best move's bar.
    probs_ax.bar([str(move) for move in board.moves], policy, color=bar_colors)
    probs_ax.set_title(f"Best Moves (State Value: {value:.3f})")
    probs_ax.set_xlabel("Moves")
    probs_ax.set_ylabel("Policy Probability")


def plot_grid_layer(layer_output, board, title):
    # Determine subplots shape.
    batch_idx, n_filters, rows, cols = layer_output.shape
    plt_rows = plt_cols = int(math.sqrt(n_filters))

    # Plot filters.
    fig = plt.figure()
    for filter_idx in range(n_filters):
        # Process filter tensor to matrix of probabilities.
        filter_activations = layer_output[0, filter_idx, ...].cpu().view(-1)  # Flatten for softmax.
        total_activation = filter_activations.abs().sum()  # Magnitude of filter's activation.
        filter_activations = filter_activations.softmax(axis=0)
        plt_data = filter_activations.view(3, 3)  # Cast back to grid shape.
        # Plot filter heatmap and overlay board state.
        plt.subplot(plt_rows, plt_cols, filter_idx + 1)
        im = plt.imshow(plt_data, cmap="coolwarm", extent=(-0.5, 2.5, 2.5, -0.5))
        for row in range(board.dim[0]):
            for col in range(board.dim[1]):
                symbol = ""
                if board.state[row][col] == board.p1.val:
                    symbol = board.p1.symbol
                if board.state[row][col] == board.p2.val:
                    symbol = board.p2.symbol
                plt.text(col, row, symbol, ha="center", va="center", fontsize=20)
        # Formatting for subplots.
        plt.title(f"Filter {filter_idx} (Activation: {total_activation:.2f})")
        plt.xticks(np.arange(-0.5, 3.0, 1))
        plt.yticks(np.arange(-0.5, 3.0, 1))
        plt.grid(which="major", color="black", linestyle="-", linewidth=2)
        plt.tick_params(axis="both", which="both", length=0)
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])

    # Formatting for main plot.
    plt.subplots_adjust(hspace=0.4)
    fig.colorbar(im, ax=fig.axes, shrink=0.8)
    fig.suptitle(f"{title} ({board.next_player.symbol} to play)")
    plt.show()
