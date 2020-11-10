#!/usr/local/bin/python3
# /usr/bin/python3
# Set the path to your python3 above

from gtp_connection import GtpConnection
from board_util import GoBoardUtil
from board import GoBoard
import random


class Gomoku():
    def __init__(self):
        """
        Gomoku player that selects moves randomly from the set of legal moves.
        Passes/resigns only at the end of the game.

        Parameters
        ----------
        name : str
            name of the player (used by the GTP interface).
        version : float
            version number (used by the GTP interface).
        """
        self.name = "GomokuAssignment3"
        self.version = 1.0
        self.NN = 10

    def get_move(self, board, color):
        return GoBoardUtil.generate_random_move(board, color)
    def get_order(self,board,color):
        move_list = board.Win()
        if move_list != []:
            return move_list
        move_list = board.BlockWin()
        if move_list != []:
            return move_list
        move_list = board.OpenFour()
        if move_list != []:
            return move_list
        move_list = board.BlockOpenFour()
        if move_list != []:
            return move_list
        move_list = board.get_empty_points().tolist()
        if move_list != []:
            return move_list

    def simulation(self, board, color,policy):
        if policy == "random":

            legal_moves = board.get_empty_points()
            score = [0]*len(legal_moves)
            for i in range(len(legal_moves)):
                temp = board.copy()
                move = legal_moves[i]

                for j in range(self.NN):
                    temp.play_move(move,color)
                    win = self.sim_for_one(temp,color)
                    if win == 1:
                        score[i] += 1
                score[i] = score[i]/self.NN
            bestIndex = score.index(max(score))
            best = legal_moves[bestIndex]
            return best
        elif policy == "rule_based":
            legal_moves = self.get_order(board, color)
            score = [0]*len(legal_moves)
            for i in range(len(legal_moves)):
                temp = board.copy()
                move = legal_moves[i]

                for j in range(self.NN):
                    temp.play_move(move,color)
                    win = self.sim_for_one_rule(temp,color)
                    if win == 1:
                        score[i] += 1
                score[i] = score[i]/self.NN
            bestIndex = score.index(max(score))
            best = legal_moves[bestIndex]
            return best


    def sim_for_one(self,board,color):

        result = board.detect_five_in_a_row()
        if result == color:
            return 1
        elif result == GoBoardUtil.opponent(color):
            return -1
        moves = GoBoardUtil.generate_random_move(board,board.current_player)
        if moves == None:
            return 0
        board.play_move(move,board.current_player)
        return self.sim_for_one(board,color)
    def sim_for_one_rule(self,board,color):

        result = board.detect_five_in_a_row()
        if result == color:
            return 1
        elif result == GoBoardUtil.opponent(color):
            return -1
        moves = self.get_order(board,board.current_player)
        if moves == None:
            return 0
        move = random.choice(moves)
        board.play_move(move,board.current_player)
        return self.sim_for_one_rule(board,color)



def run():
    """
    start the gtp connection and wait for commands.
    """
    board = GoBoard(7)
    con = GtpConnection(Gomoku(), board)
    con.start_connection()


if __name__ == "__main__":
    run()
