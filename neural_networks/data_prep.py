import numpy as np
import torch

from tic_tac_toe.board import TicTacToeBoard


def _board_to_planes(board: TicTacToeBoard):
    # Plane representing last player's moves.
    state_last = board.state == board.last_player.val
    # Plane representing current player's moves.
    state_curr = board.state == board.next_player.val
    # Plane representing the remaining moves on the board.
    state_neutral = board.state == 0
    return (state_last, state_neutral, state_curr)


def get_input_feats(board: TicTacToeBoard, device: torch.device = None) -> torch.Tensor:
    """
    Create feature plane stack from the raw board state.
    """
    planes = _board_to_planes(board)
    encoded_state = np.stack(planes).astype(np.float32)
    # Re-add batch dimension here as it is null.
    return torch.tensor(encoded_state, device=device).unsqueeze(0)


def get_batch_input_feats(boards: TicTacToeBoard, device: torch.device = None) -> torch.Tensor:
    planes = [_board_to_planes(board) for board in boards]
    encoded_states = np.stack(planes).astype(np.float32)
    return torch.tensor(encoded_states, device=device)


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


def apply_dirichlet_noise(policy: np.ndarray, eps: float, alpha: float) -> np.ndarray:
    return (1 - eps) * policy + eps * np.random.dirichlet([alpha] * 9)
