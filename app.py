import time
from threading import Event, Thread

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from tic_tac_toe.board import TicTacToeBoard
from tic_tac_toe.mcts import MCSTNode
from tic_tac_toe.parse_tree import parse_mcst
from tic_tac_toe.player import TicTacToeBot


class MCTSApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.thread = Thread(target=self.background_thread)
        self.thread_stop_event = Event()
        self.tree_data = self.generate_tree_data()

        @self.app.route("/")
        def index():
            return render_template("index.html")

        @self.socketio.on("connect")
        def handle_connect():
            emit("message", {"data": "Connected to server"})

        @self.socketio.on("update_request")
        def handle_update_request(json):
            if not self.thread.is_alive():
                self.thread = Thread(target=self.background_thread)
                self.thread.start()

    def generate_tree_data(self):
        opts = {"sim_lim": 9 * 10, "c": 1.4, "win": 1.0, "lose": -1.0, "draw": 0.5}
        board = TicTacToeBoard(TicTacToeBot("p1", "X", "random"), TicTacToeBot("p2", "O", "random"))
        root_node = MCSTNode(None, board, opts)
        root_node.sim()  # TODO: add sleep between sim iters.
        # NOTE: how do I get the data at each step out? Generator? can do below for now.
        return parse_mcst(root_node)

    def background_thread(self):
        while not self.thread_stop_event.is_set():
            self.socketio.emit("update_response", self.tree_data)
            time.sleep(1)  # TODO: SLEEP DURING GENERATION INSTEAD LA

    def run(self):
        self.socketio.run(self.app, debug=True)


if __name__ == "__main__":
    mcts_app = MCTSApp()
    mcts_app.run()
