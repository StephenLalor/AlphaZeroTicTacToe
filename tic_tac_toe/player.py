from typing import Literal


class TicTacToeBot:
    def __init__(self, name: str, symbol: Literal["X", "O"]):
        """
        TicTacToe player, storing player name, symbol and value associated with symbol.
        """
        if symbol not in ("X", "O"):
            raise ValueError("Symbol not 'X' or 'O'")
        self.name = name
        self.symbol = symbol  # Display only.
        self.val = 1 if self.symbol == "X" else 2

    def __repr__(self):
        return f"TicTacToeBot(name={self.name}, symbol={self.symbol})"
