import mlflow
import torch

from neural_networks.data_prep import get_input_feats
from neural_networks.tic_tac_toe_net import TicTacToeNet
from tic_tac_toe.board import TicTacToeBoard
from tic_tac_toe.player import TicTacToeBot


def validate_model_inference(mdl: TicTacToeNet, board: TicTacToeBoard = None) -> None:
    """
    Check model can do prediction, raise if output structure incorrect.
    """
    # Check model in inference mode.
    if mdl.training:
        raise RuntimeError("Model not in inference mode")

    # Create sample board if needed and test inference.
    if not board:
        board = TicTacToeBoard(TicTacToeBot("p1", "X"), TicTacToeBot("p2", "O"))
    policy, value = mdl(get_input_feats(board))

    # Check output structure.
    exp_action_size = board.dim[0] * board.dim[0]
    exp_pol_shape, exp_val_shape = (1, exp_action_size), (1, 1)
    if policy.shape != exp_pol_shape:
        raise ValueError(f"Expected shape {exp_pol_shape} but got {policy.shape} for policy.")
    if value.shape != exp_val_shape:
        raise ValueError(f"Expected shape {exp_val_shape} but got {value.shape} for value.")


def load_mdl_for_inference(mdl_name: str, mdl_ver: int) -> TicTacToeNet:
    """
    Load and configure model for inference, then validate.
    """
    mdl = mlflow.pytorch.load_model(f"models:/{mdl_name}/{mdl_ver}")
    mdl.eval()
    if torch.cuda.is_available():
        mdl.to(torch.device("cuda"))
    validate_model_inference(mdl)
    return mdl
