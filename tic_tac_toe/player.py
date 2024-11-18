import random


class TicTacToeBot:
    def __init__(self, name, symbol, strategy):
        self.name = name
        self.symbol = symbol
        self.strategy = strategy

    def __str__(self):
        return f"Name: {self.name}, Sym: {self.symbol}, Strat: {self.strategy}"

    def generate_move(self, valid_moves):
        if self.strategy == "random":
            move = self.generate_random_move(valid_moves)
        return move

    def generate_random_move(self, valid_moves):
        return (random.choice(valid_moves), self.symbol)  # TODO: Seed this.
