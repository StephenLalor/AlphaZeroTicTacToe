"""
Utility script to train model.
"""

from self_play.self_play import TicTacToeTrainer
from utils import read_config, set_up_logging

if __name__ == "__main__":
    # Setup.
    set_up_logging()
    cfg = read_config("training_cfg.yaml")  # Choose config script here.

    # Train with self play.
    trainer = TicTacToeTrainer(cfg)
    trainer.self_train()
