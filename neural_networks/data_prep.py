import numpy as np
import torch

from tic_tac_toe.board import TicTacToeBoard


def get_input_feats(board: TicTacToeBoard) -> torch.Tensor:
    """
    Create feature plane stack from the raw board state.
    """
    # Plane representing last player's moves.
    state_last = board.state == board.last_player.val
    # Plane representing current player's moves.
    state_curr = board.state == board.next_player.val
    # Plane representing the remaining moves on the board.
    state_neutral = board.state == 0
    # Stack into [3, 3, 3] tensor.
    encoded_state = np.stack((state_last, state_neutral, state_curr)).astype(np.float32)
    return torch.tensor(encoded_state).unsqueeze(0)


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
    return {move: prob for move, prob in zip(all_moves, masked_policy.tolist())}
