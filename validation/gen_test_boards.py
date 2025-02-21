from tic_tac_toe.board import TicTacToeBoard
from tic_tac_toe.player import TicTacToeBot


def gen_test_boards() -> list[TicTacToeBoard]:
    """
    Generate several interesting board states.
    """

    # Sequences of moves to construct each board.
    move_seqs = [
        [(0, 0), (1, 0), (2, 1), (1, 1), (2, 0)],
        [(0, 0), (2, 2), (0, 1), (1, 1)],
        [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)],
        [(0, 0), (0, 1), (1, 1), (0, 2)],
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1)],
    ]

    # Build each board.
    boards = []
    for move_seq in move_seqs:
        # Execute each move in sequence to get final board.
        board = TicTacToeBoard(TicTacToeBot("p1", "X"), TicTacToeBot("p2", "O"))
        for move in move_seq:
            board.exec_move(move)
        boards.append(board)

    return boards
