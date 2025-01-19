import numpy as np
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
    last_player_plane = state == board.last_player.val
    next_player_plane = state == board.next_player.val
    # Create to-play plane.
    if board.next_player == board.p1:
        to_play_plane = torch.ones(board.dim, dtype=torch.float32)
    else:
        to_play_plane = torch.zeros(board.dim, dtype=torch.float32)
    # Stack planes to create input features.
    stack = torch.stack([last_player_plane, next_player_plane, to_play_plane])
    # Add batch dimension.
    return stack.unsqueeze(0)


def policy_to_valid_moves(policy: np.ndarray, all_moves: list, valid_moves: list) -> dict:
    """
    Set probabilities for invalid moves in the policy to zero and return a mapping from move to the
    move's probability.
    """
    # Mask invalid moves so they have probability of zero.
    moves_mask = [1 if move in valid_moves else 0 for move in all_moves]
    masked_policy = moves_mask * policy
    masked_policy = masked_policy / np.sum(masked_policy)  # Re-norm probabilities.
    # Map moves to their probabilities.
    return {move: prob for move, prob in zip(all_moves, masked_policy.tolist()) if prob > 0}
