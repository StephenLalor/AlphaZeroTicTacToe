from tic_tac_toe.board import TicTacToeException
from tic_tac_toe.player import TicTacToeBot


def assign_reward(player: TicTacToeBot, last_player: TicTacToeBot, res: str, rewards: dict):
    """
    Assign reward for game termination based on who played last.
    """
    # Ensure only handled results are passed.
    allowed_results = {"win", "draw"}
    if res not in allowed_results:
        raise TicTacToeException(f"Result {res} not one of {allowed_results}")
    # Ensure rewards has all required levels.
    missing_keys = {"win", "draw", "lose"} - set(rewards)
    if missing_keys:
        raise TicTacToeException(f"Rewards dictionary missing keys: {missing_keys}")
    # Determine reward with respect to player.
    if res == "draw":
        return rewards["draw"]
    if res == "win" and player == last_player:
        return rewards["win"]
    return rewards["lose"]
