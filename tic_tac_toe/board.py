import numpy as np

from tic_tac_toe.player import TicTacToeBot


class TicTacToeException(Exception):
    pass


class TicTacToeBoard:
    def __init__(self, p1: TicTacToeBot, p2: TicTacToeBot):
        # Set up 3x3 board and define value move locations.
        self.dim = (3, 3)
        self.state = np.zeros(self.dim, dtype=np.int8)  # Game state i.e. the actual board.
        self.moves = [loc for loc in np.ndindex(self.state.shape)]  # All possible moves.
        self.valid_moves = self.moves.copy()  # Currently possible moves.
        # Set up players, defaulting to p1 being next to play.
        self.p1 = p1
        self.p2 = p2
        self.next_player = p1
        self.last_player = p2
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

    def update_result(self):
        """
        Check if current board state is terminal.
        """
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
        """
        Execute move at move_loc for the current player.
        """
        # Check game is still in play.
        if self.game_result:
            raise TicTacToeException("Game already ended.")
        # Check move is valid.
        if move_loc not in self.valid_moves:
            raise TicTacToeException(f"{move_loc} is not valid move for this board.")
        # Next player makes their move.
        self.state[move_loc] = self.next_player.val
        self.valid_moves.remove(move_loc)
        # Rotate players so player who just played is last_player.
        self.next_player, self.last_player = self.last_player, self.next_player
        self.last_move = move_loc
        # Check if this resulted in a win or draw.
        self.update_result()
        return
