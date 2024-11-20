import time
from threading import Event, Thread

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from mcts.mcts import MCSTNode
from mcts.parse_tree import parse_mcst
from tic_tac_toe.board import TicTacToeBoard
from tic_tac_toe.player import TicTacToeBot


class TicTacToeBotApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.thread = None
        self.thread_stop_event = Event()

        @self.app.route("/")
        def index():
            return render_template("index.html")

        @self.socketio.on("connect")
        def handle_connect():
            emit("message", {"data": "Connected to server"})

        @self.socketio.on("update_request")
        def handle_update_request(json):
            if not self.thread or not self.thread.is_alive():
                self.thread = Thread(target=self.generate_tree_data)
                self.thread.start()

    def generate_tree_data(self):
        opts = {
            "sim_lim": (9**3) + 1,
            "c": 1.4,
            "win": 1.0,
            "lose": -1.0,
            "draw": 0.5,
        }
        p1 = TicTacToeBot("p1", 1, "random")
        p2 = TicTacToeBot("p2", 2, "random")
        board = TicTacToeBoard(p1, p2)
        root_node = MCSTNode(None, board, opts)
        for root in root_node.sim():
            update_data = parse_mcst(root)
            self.socketio.emit("update_response", update_data)
            time.sleep(0.2)
        self.socketio.emit("final_update", update_data)  # Emit the final state

    def run(self):
        self.socketio.run(self.app, debug=True)


if __name__ == "__main__":
    mcts_app = TicTacToeBotApp()
    mcts_app.run()
