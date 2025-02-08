from collections import Counter

import torch

from self_play.game_data import GameDataset

# Load games.
game_dataset = torch.load("validation/data/game_dataset_3_games.pt", weights_only=False)


def get_game_patterns(game_dataset: GameDataset):
    move_counts, pattern_counts = Counter(), Counter()
    for game in game_dataset.games:
        pattern = tuple([(int(move[0]), int(move[1])) for move in game.moves])
        move_counts.update(pattern)
        pattern_counts.update([pattern])
    return move_counts, pattern_counts


move_counts, pattern_counts = get_game_patterns(game_dataset)

# BUG: POLICY OUT OF ORDER HERE!!!
game = game_dataset.games[0]
turn = -1
game.states[turn]
game.pol_targets[turn]
game.moves[turn]
game.rewards[turn]
