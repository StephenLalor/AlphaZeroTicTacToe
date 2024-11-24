# TODO: This should probably not be here, and be a method instead.

from mcts.brute_mcts import BruteMCSTNode


def parse_mcst_node(node: BruteMCSTNode) -> dict:
    parsed_node = {
        "visits": node.visits if node.visits else None,
        "value": node.value if node.value else None,
        "player": node.board.last_player.name if node.board.last_player else None,
        "move": node.board.last_move if node.board.last_move else None,
        "board": node.board.state.tolist(),
    }
    return parsed_node


def parse_mcst(root: BruteMCSTNode) -> dict[list, list]:
    """
    Parse Monte Carlo Search Tree into graphable format.
    """

    def traverse(node, data, parent_id):
        """
        Recursive helper function to traverse the tree.
        """
        # Return at leaf nodes.
        if not node:
            return data
        # Uniquely identify node and store data for visualisation.
        node_id = id(node)
        data["nodes"].append({"id": node_id, "label": parse_mcst_node(node)})
        data["edges"].append({"from": parent_id, "to": node_id} if parent_id else None)
        # Recurse into each child node.
        for child in node.children:
            data = traverse(child, data, parent_id=node_id)
        return data

    return traverse(node=root, data={"nodes": [], "edges": []}, parent_id=None)
