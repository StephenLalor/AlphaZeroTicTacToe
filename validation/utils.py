# TODO: Move to proper location and clean up.

import numpy as np
import torch

from neural_networks.data_prep import get_input_feats


def get_policy_and_value(board, mdl):
    with torch.no_grad():
        policy, value = mdl(get_input_feats(board, mdl.device))
    policy, value = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy(), value.item()
    best_move = board.moves[np.argmax(policy)]
    return policy, value, best_move


def get_layer_output(layer_output, name):
    def hook(mdl, nn_in, nn_out):
        layer_output[name] = nn_out.detach()

    return hook
