import numpy as np

from tic_tac_toe.player import TicTacToeBot


class TicTacToeException(Exception):
    pass


class TicTacToeBoard:
    def __init__(self, p1: TicTacToeBot, p2: TicTacToeBot):
        self.p1 = p1
        self.p2 = p2
        self.last_player = None
        self.dim = None
        self.state = None
        self.valid_moves = None
        self.last_move = None
        self.game_result = None
        self.init_board()

    def __str__(self):
        if self.state is None:
            return ""
        lst = self.state.tolist()
        parsed_board = [[f"[{x.decode() if x.decode() else " "}]" for x in row] for row in lst]
        formatted_board = "\n".join([" ".join(row) for row in parsed_board])
        header_msg = f"Board State (moves: {len(self.valid_moves)}, status: {str(self.game_result)})"
        final_msg = f"{header_msg}\n{formatted_board}"
        return final_msg

    def init_board(self):
        self.dim = (3, 3)
        self.state = np.zeros(self.dim, dtype="S1")
        self.valid_moves = [loc for loc in np.ndindex(self.state.shape)]  # List as we will make random selection later.
        self.game_result = None
        self.last_player = None
        self.last_move = None

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
        # Co-ords for diagonal and ancy indexing to avoid flipping state.
        diag = self.state[list(row_diag), list(col_diag)]
        row_adiag, col_adiag = zip(*[(0 + i, self.dim[1] - 1 - i) for i in range(self.dim[1])])
        # Co-ords for anti-diagonal and fancy indexing.
        anti_diag = self.state[list(row_adiag), list(col_adiag)]
        if anti_diag[0] and np.all(anti_diag == anti_diag[0]) or (diag[0] and np.all(diag == diag[0])):
            self.game_result = "win"
            return

        # Check if draw.
        if not self.valid_moves:  # Not a win but no valid moves available.
            self.game_result = "draw"
        return

    def get_next_player(self):
        return self.p1 if self.last_player in [self.p2, None] else self.p2

    def exec_move(self, move_loc):
        player = self.get_next_player()  # We need the right symbol.
        self.update_board(player.symbol, move_loc)
        self.last_player = player
        self.last_move = move_loc
        self.update_result()
        return
