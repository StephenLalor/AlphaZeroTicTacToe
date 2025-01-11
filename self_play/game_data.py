import numpy as np
import torch
from torch.utils.data import DataLoader

from neural_networks.data_prep import get_input_feats
from tic_tac_toe.board import TicTacToeBoard, assign_reward
from tic_tac_toe.player import TicTacToeBot


class GameData:
    def __init__(self):
        self.states = []
        self.pol_targets = []
        self.rewards = []
        self.players = []

    def append_turn(self, board: TicTacToeBoard, probs: np.array, player: TicTacToeBot):
        self.states.append(get_input_feats(board).squeeze(0))
        self.pol_targets.append(torch.from_numpy(probs))
        self.players.append(player)

    def finalise(self, last: TicTacToeBot, res: TicTacToeBot | str, rewards_cfg: dict):
        # Add reward for each turn.
        for turn in range(len(self.players)):
            reward = assign_reward(last, self.players[turn], res, rewards_cfg)
            self.rewards.append(torch.tensor(reward, dtype=torch.float32))

        # Finalise data by stacking into tensors.
        self.states = torch.stack(self.states)
        self.pol_targets = torch.stack(self.pol_targets)
        self.rewards = torch.stack(self.rewards)

    def to(self, device: str):
        self.states = self.states.to(device)
        self.pol_targets = self.pol_targets.to(device)
        self.rewards = self.rewards.to(device)


class GameDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.games = []

    def __len__(self):
        return len(self.games)

    def __getitem__(self, i: int):
        return self.games[i]

    def append_game(self, game: GameData):
        self.games.append(game)

    def to(self, device: str):
        for game in self.games:
            game.to(device)

    def _collate(self, batch: list[GameData]):
        """
        Split game data into tensors of game states, policy targets and value targets (rewards).
        """
        batch_states = torch.cat([game.states for game in batch], dim=0)
        batch_pol_targets = torch.cat([game.pol_targets for game in batch], dim=0)
        batch_rewards = torch.cat([game.rewards for game in batch], dim=0)
        return batch_states, batch_pol_targets, batch_rewards

    def to_loader(self, batch_size: int) -> DataLoader:
        """
        Convert game data to batch loader with shuffling.
        """
        return DataLoader(self, batch_size, shuffle=True, collate_fn=self._collate)
