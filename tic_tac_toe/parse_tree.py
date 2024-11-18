from tic_tac_toe.mcts import MCSTNode


def parse_mcst_node(node: MCSTNode) -> dict:
    # TODO: Parse the board state as well in a way that can be plotted.
    parsed_node = {
        "visits": node.visits,
        "value": node.value,
        "player": node.board.last_player.name,
        "move": node.board.last_move,
    }
    return parsed_node


def parse_mcst(root: MCSTNode) -> dict[list, list]:
    """
    Parse Monte Carlo Search Tree into graphable format.
    """

    def traverse(node: MCSTNode, data: dict[list, list], parent_id: int) -> dict[list, list]:
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
