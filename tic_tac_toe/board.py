import numpy as np

from tic_tac_toe.player import TicTacToeBot


class TicTacToeException(Exception):
    pass


class TicTacToeBoard:
    def __init__(self, p1: TicTacToeBot, p2: TicTacToeBot):
        self.dim = (3, 3)
        self.state = np.zeros(self.dim, dtype=np.int8)
        self.moves = [loc for loc in np.ndindex(self.state.shape)]
        self.valid_moves = self.moves.copy()
        self.p1 = p1
        self.p2 = p2
        self.next_player = p1
        self.last_player = p2
        self.last_move = None
        self.game_result = None

    def __str__(self):
        if self.state is None:
            return ""
        formatted_board = "\n".join(["".join(str(row)) for row in self.state.tolist()])
        header_msg = (
            f"Board State (moves: {len(self.valid_moves)}, status: {str(self.game_result)})"
        )
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
        self.update_board(self.next_player.symbol, move_loc)
        self.next_player, self.last_player = self.last_player, self.next_player
        self.last_move = move_loc
        self.update_result()
        return
