import torch

from tic_tac_toe.board import TicTacToeBoard


def get_input_feats(board: TicTacToeBoard) -> torch.Tensor:
    """
    Create feature plane stack with dims [batch_size, channels, height, width] consisting of the
    current player feature plane, next player feature plane and a plane indicating which player is
    to play next.

    There is no need for historical context in TicTacToe so only one plane for each player's
    position is needed.
    """
    # TODO: Use tensor for board state to begin with.
    state = torch.from_numpy(board.state)
    # Create player planes.
    last_player_plane = state == board.last_player.symbol
    next_player_plane = state == board.next_player.symbol
    # Create to-play plane.
    if board.next_player == board.p1:
        to_play_plane = torch.ones(board.dim)
    else:
        to_play_plane = torch.zeros(board.dim)
    # Stack planes to create input features.
    stack = torch.stack([last_player_plane, next_player_plane, to_play_plane])
    # Add batch dimension.
    return stack.unsqueeze(0)
