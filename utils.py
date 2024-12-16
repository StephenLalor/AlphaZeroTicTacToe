import logging
import logging.config
from collections import deque

import yaml

from mcts.alpha_zero_mcts import SmartMCSTNode


def set_up_logging(default_path="logging_config.yaml", default_level=logging.INFO):
    with open(default_path, "rt") as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)


def read_config(path):
    with open(path, "rt") as f:
        cfg = yaml.safe_load(f.read())
    return cfg


def print_tree(root: SmartMCSTNode) -> None:
    """
    Print the tree nodes in level order.
    """
    level = 0  # Current level of the tree.
    queue = deque([root])
    while queue:
        # Build current level.
        level_nodes = []
        for _ in range(len(queue)):
            # Add next in queue to current level.
            node = queue.popleft()
            level_nodes.append(node)
            # Queue up all child nodes.
            for child in node.children:
                queue.append(child)
        # Print completed level nodes and info.
        level += 1
        num_nodes = str(len(level_nodes))
        print(f"------- [Level: {str(level)}] [Nodes: {num_nodes}] -------")
        for node in level_nodes:
            print(node)
    return
