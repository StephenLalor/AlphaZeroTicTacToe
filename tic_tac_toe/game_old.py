import copy

# TODO: Maybe try some async to play games simultaneously.
class TicTacToeGame:
    def __init__(self, player_1, player_2, board, turn_lim=100, match_limit=3):
        self.player_1 = player_1
        self.player_2 = player_2
        self.board = board
        self.turn_lim = turn_lim
        self.match_limit = match_limit
        self.match_hist = {}

    def gen_turn_hist(self, player=None, board=None, move=None):
        game_hist = {
            "state": copy.deepcopy(board.state) if board else None,  # Deep needed as nested list.
            "player": player,
            "move": move,
            "result": board.game_result if board else None
        }
        return game_hist

    def sim(self, record_hist):
        match_num = 0
        # Play matches until we reach the match limit.
        while match_num < self.match_limit:
            # Initialise match.
            match_num += 1
            turn_num, turn_hist = 0, {}
            self.board.init_board()  # Reset board for re-use.
            if record_hist:
                turn_hist[turn_num] = self.gen_turn_hist()
            # Execute moves until game is over or turn_num limit hit.
            while turn_num < self.turn_lim and not self.board.game_result:
                for player in [self.player_1, self.player_2]:
                    # Terminate game if it's over.
                    if self.board.game_result:
                        if record_hist:
                            turn_hist["res"] = f"Result: {player.name} {self.board.game_result}"  # TODO: May need winner regardless.
                        break
                    # Generate and execute the player's move.
                    turn_num += 1
                    move = player.generate_move(self.board.valid_moves)
                    self.board.move(move[0], move[1])
                    # Add move to history.
                    if record_hist:
                        turn_hist[turn_num] = self.gen_turn_hist(player, self.board, move)
            self.match_hist[match_num] = copy.deepcopy(turn_hist)

        return
