import numpy as np

from tic_tac_toe.player import TicTacToeBot


class TicTacToeException(Exception):
    pass


class TicTacToeBoard:
    def __init__(
        self,
        p1: TicTacToeBot,
        p2: TicTacToeBot,
        last_player: TicTacToeBot = None,
        next_player: TicTacToeBot = None,
    ):
        # Set up 3x3 board and define value move locations.
        self.dim = (3, 3)
        self.state = np.zeros(self.dim, dtype=np.int8)
        self.moves = [loc for loc in np.ndindex(self.state.shape)]
        self.valid_moves = self.moves.copy()
        # Set up players, defaulting to p1 being next to play.
        self.p1 = p1
        self.p2 = p2
        self.next_player = next_player if next_player else p1
        self.last_player = last_player if last_player else p2
        # Track game results and the last executed move on the board.
        self.last_move = None
        self.game_result = None

    def __str__(self):
        if self.state is None:
            return ""
        formatted_board = "\n".join(["".join(str(row)) for row in self.state.tolist()])
        status = "terminal" if self.game_result else "active"
        header_msg = f"Board State (moves: {len(self.valid_moves)}, status: {status})"
        return f"{header_msg}\n{formatted_board}"

    def update_board(self, symbol, move_loc):
        self.state[move_loc] = symbol
        self.valid_moves.remove(move_loc)
        return

    def update_result(self):
        # Check each row for a win.
        for i in range(self.dim[0]):
            if self.state[(i, 0)] and np.all(self.state[i] == self.state[(i, 0)]):
                self.game_result = "win"
                return
        # Check each col for a win.
        for j in range(self.dim[1]):
            if self.state[(0, j)] and np.all(self.state[:, j] == self.state[(0, j)]):
                self.game_result = "win"
                return

        # Check for diagonal win.
        row_diag, col_diag = zip(*[(0 + i, 0 + i) for i in range(self.dim[0])])
        row_adiag, col_adiag = zip(*[(0 + i, self.dim[1] - 1 - i) for i in range(self.dim[1])])
        # Co-ords for diagonal and fancy indexing to avoid flipping state.
        diag = self.state[list(row_diag), list(col_diag)]
        adiag = self.state[list(row_adiag), list(col_adiag)]
        # Check for winning state.
        adiag_win = adiag[0] and np.all(adiag == adiag[0])
        diag_win = diag[0] and np.all(diag == diag[0])
        if adiag_win or diag_win:
            self.game_result = "win"
            return

        # Check if draw.
        if not self.valid_moves:  # Not a win but no valid moves available.
            self.game_result = "draw"
        return

    def exec_move(self, move_loc):
        # Next player makes their move.
        self.update_board(self.next_player.symbol, move_loc)
        # Rotate players so player who just played is last_player.
        self.next_player, self.last_player = self.last_player, self.next_player
        self.last_move = move_loc
        # Check if this resulted in a win or draw.
        self.update_result()
        return


def assign_reward(player: TicTacToeBot, last_player: TicTacToeBot, res: str, rewards: dict):
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
