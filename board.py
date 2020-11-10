"""
board.py

Implements a basic Go board with functions to:
- initialize to a given board size
- check if a move is legal
- play a move

The board uses a 1-dimensional representation with padding
"""

import numpy as np
from board_util import (
    GoBoardUtil,
    BLACK,
    WHITE,
    EMPTY,
    BORDER,
    PASS,
    is_black_white,
    is_black_white_empty,
    coord_to_point,
    where1d,
    MAXSIZE,
    GO_POINT
)

"""
The GoBoard class implements a board and basic functions to play
moves, check the end of the game, and count the acore at the end.
The class also contains basic utility functions for writing a Go player.
For many more utility functions, see the GoBoardUtil class in board_util.py.

The board is stored as a one-dimensional array of GO_POINT in self.board.
See GoBoardUtil.coord_to_point for explanations of the array encoding.
"""
class GoBoard(object):
    def __init__(self, size):
        """
        Creates a Go board of given size
        """
        assert 2 <= size <= MAXSIZE
        self.reset(size)
        self.calculate_rows_cols_diags()

    def calculate_rows_cols_diags(self):
        if self.size < 5:
            return
        # precalculate all rows, cols, and diags for 5-in-a-row detection
        self.rows = []
        self.cols = []
        for i in range(1, self.size + 1):
            current_row = []
            start = self.row_start(i)
            for pt in range(start, start + self.size):
                current_row.append(pt)
            self.rows.append(current_row)

            start = self.row_start(1) + i - 1
            current_col = []
            for pt in range(start, self.row_start(self.size) + i, self.NS):
                current_col.append(pt)
            self.cols.append(current_col)

        self.diags = []
        # diag towards SE, starting from first row (1,1) moving right to (1,n)
        start = self.row_start(1)
        for i in range(start, start + self.size):
            diag_SE = []
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_SE.append(pt)
                pt += self.NS + 1
            if len(diag_SE) >= 5:
                self.diags.append(diag_SE)
        # diag towards SE and NE, starting from (2,1) downwards to (n,1)
        for i in range(start + self.NS, self.row_start(self.size) + 1, self.NS):
            diag_SE = []
            diag_NE = []
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_SE.append(pt)
                pt += self.NS + 1
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_NE.append(pt)
                pt += -1 * self.NS + 1
            if len(diag_SE) >= 5:
                self.diags.append(diag_SE)
            if len(diag_NE) >= 5:
                self.diags.append(diag_NE)
        # diag towards NE, starting from (n,2) moving right to (n,n)
        start = self.row_start(self.size) + 1
        for i in range(start, start + self.size):
            diag_NE = []
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_NE.append(pt)
                pt += -1 * self.NS + 1
            if len(diag_NE) >=5:
                self.diags.append(diag_NE)
        assert len(self.rows) == self.size
        assert len(self.cols) == self.size
        assert len(self.diags) == (2 * (self.size - 5) + 1) * 2

    def reset(self, size):
        """
        Creates a start state, an empty board with given size.
        """
        self.size = size
        self.NS = size + 1
        self.WE = 1
        self.ko_recapture = None
        self.last_move = None
        self.last2_move = None
        self.current_player = BLACK
        self.maxpoint = size * size + 3 * (size + 1)
        self.board = np.full(self.maxpoint, BORDER, dtype=GO_POINT)
        self._initialize_empty_points(self.board)
        self.calculate_rows_cols_diags()

    def copy(self):
        b = GoBoard(self.size)
        assert b.NS == self.NS
        assert b.WE == self.WE
        b.ko_recapture = self.ko_recapture
        b.last_move = self.last_move
        b.last2_move = self.last2_move
        b.current_player = self.current_player
        assert b.maxpoint == self.maxpoint
        b.board = np.copy(self.board)
        return b

    def get_color(self, point):
        return self.board[point]

    def pt(self, row, col):
        return coord_to_point(row, col, self.size)

    def is_legal(self, point, color):
        """
        Check whether it is legal for color to play on point
        This method tries to play the move on a temporary copy of the board.
        This prevents the board from being modified by the move
        """
        board_copy = self.copy()
        can_play_move = board_copy.play_move(point, color)
        return can_play_move

    def get_empty_points(self):
        """
        Return:
            The empty points on the board
        """
        return where1d(self.board == EMPTY)

    def get_color_points(self, color):
        """
        Return:
            All points of color on the board
        """
        return where1d(self.board == color)

    def row_start(self, row):
        assert row >= 1
        assert row <= self.size
        return row * self.NS + 1

    def _initialize_empty_points(self, board):
        """
        Fills points on the board with EMPTY
        Argument
        ---------
        board: numpy array, filled with BORDER
        """
        for row in range(1, self.size + 1):
            start = self.row_start(row)
            board[start : start + self.size] = EMPTY

    def is_eye(self, point, color):
        """
        Check if point is a simple eye for color
        """
        if not self._is_surrounded(point, color):
            return False
        # Eye-like shape. Check diagonals to detect false eye
        opp_color = GoBoardUtil.opponent(color)
        false_count = 0
        at_edge = 0
        for d in self._diag_neighbors(point):
            if self.board[d] == BORDER:
                at_edge = 1
            elif self.board[d] == opp_color:
                false_count += 1
        return false_count <= 1 - at_edge  # 0 at edge, 1 in center

    def _is_surrounded(self, point, color):
        """
        check whether empty point is surrounded by stones of color
        (or BORDER) neighbors
        """
        for nb in self._neighbors(point):
            nb_color = self.board[nb]
            if nb_color != BORDER and nb_color != color:
                return False
        return True

    def _has_liberty(self, block):
        """
        Check if the given block has any liberty.
        block is a numpy boolean array
        """
        for stone in where1d(block):
            empty_nbs = self.neighbors_of_color(stone, EMPTY)
            if empty_nbs:
                return True
        return False

    def _block_of(self, stone):
        """
        Find the block of given stone
        Returns a board of boolean markers which are set for
        all the points in the block
        """
        color = self.get_color(stone)
        assert is_black_white(color)
        return self.connected_component(stone)

    def connected_component(self, point):
        """
        Find the connected component of the given point.
        """
        marker = np.full(self.maxpoint, False, dtype=bool)
        pointstack = [point]
        color = self.get_color(point)
        assert is_black_white_empty(color)
        marker[point] = True
        while pointstack:
            p = pointstack.pop()
            neighbors = self.neighbors_of_color(p, color)
            for nb in neighbors:
                if not marker[nb]:
                    marker[nb] = True
                    pointstack.append(nb)
        return marker

    def _detect_and_process_capture(self, nb_point):
        """
        Check whether opponent block on nb_point is captured.
        If yes, remove the stones.
        Returns the stone if only a single stone was captured,
        and returns None otherwise.
        This result is used in play_move to check for possible ko
        """
        single_capture = None
        opp_block = self._block_of(nb_point)
        if not self._has_liberty(opp_block):
            captures = list(where1d(opp_block))
            self.board[captures] = EMPTY
            if len(captures) == 1:
                single_capture = nb_point
        return single_capture

    def play_move(self, point, color):
        """
        Play a move of color on point
        Returns boolean: whether move was legal
        """
        assert is_black_white(color)
        # Special cases
        if point == PASS:
            self.ko_recapture = None
            self.current_player = GoBoardUtil.opponent(color)
            self.last2_move = self.last_move
            self.last_move = point
            return True
        elif self.board[point] != EMPTY:
            return False
        # if point == self.ko_recapture:
        #     return False

        # General case: deal with captures, suicide, and next ko point
        # opp_color = GoBoardUtil.opponent(color)
        # in_enemy_eye = self._is_surrounded(point, opp_color)
        self.board[point] = color
        # single_captures = []
        # neighbors = self._neighbors(point)
        # for nb in neighbors:
        #     if self.board[nb] == opp_color:
        #         single_capture = self._detect_and_process_capture(nb)
        #         if single_capture != None:
        #             single_captures.append(single_capture)
        # block = self._block_of(point)
        # if not self._has_liberty(block):  # undo suicide move
        #     self.board[point] = EMPTY
        #     return False
        # self.ko_recapture = None
        # if in_enemy_eye and len(single_captures) == 1:
        #     self.ko_recapture = single_captures[0]
        self.current_player = GoBoardUtil.opponent(color)
        self.last2_move = self.last_move
        self.last_move = point
        return True

    def neighbors_of_color(self, point, color):
        """ List of neighbors of point of given color """
        nbc = []
        for nb in self._neighbors(point):
            if self.get_color(nb) == color:
                nbc.append(nb)
        return nbc

    def _neighbors(self, point):
        """ List of all four neighbors of the point """
        return [point - 1, point + 1, point - self.NS, point + self.NS]

    def _diag_neighbors(self, point):
        """ List of all four diagonal neighbors of point """
        return [
            point - self.NS - 1,
            point - self.NS + 1,
            point + self.NS - 1,
            point + self.NS + 1,
        ]

    def last_board_moves(self):
        """
        Get the list of last_move and second last move.
        Only include moves on the board (not None, not PASS).
        """
        board_moves = []
        if self.last_move != None and self.last_move != PASS:
            board_moves.append(self.last_move)
        if self.last2_move != None and self.last2_move != PASS:
            board_moves.append(self.last2_move)
            return

    def detect_five_in_a_row(self):
        """
        Returns BLACK or WHITE if any five in a row is detected for the color
        EMPTY otherwise.
        """
        for r in self.rows:
            result = self.has_five_in_list(r)
            if result != EMPTY:
                return result
        for c in self.cols:
            result = self.has_five_in_list(c)
            if result != EMPTY:
                return result
        for d in self.diags:
            result = self.has_five_in_list(d)
            if result != EMPTY:
                return result
        return EMPTY

    def has_five_in_list(self, list):
        """
        Returns BLACK or WHITE if any five in a rows exist in the list.
        EMPTY otherwise.
        """
        prev = BORDER
        counter = 1
        for stone in list:
            if self.get_color(stone) == prev:
                counter += 1
            else:
                counter = 1
                prev = self.get_color(stone)
            if counter == 5 and prev != EMPTY:
                return prev
        return EMPTY
    def checkrow(self, start,color):
        checkColor = self.board[start]
        samecolor = 1
        space = 0
        start1 = start
        while True:
            start += 1
            if self.board[start] == color:
                samecolor += 1
            else:
                if self.board[start] == EMPTY:
                    space += 1
                break
        while True:
            start1 -= 1
            if self.board[start1] == color:
                samecolor += 1
            else:
                if self.board[start1] == EMPTY:
                    space += 1
                break
        return samecolor, space
    def checkcol(self, start,color):
        checkColor = self.board[start]
        samecolor = 1
        space = 0
        start1 = start
        while True:
            start += self.NS
            if self.board[start] == color:
                samecolor += 1
            else:
                if self.board[start] == EMPTY:
                    space += 1
                break
        while True:
            start1 -= self.NS
            if self.board[start1] == color:
                samecolor += 1
            else:
                if self.board[start1] == EMPTY:
                    space += 1
                break
        return samecolor, space
    def checkd1(self, start,color):
        checkColor = self.board[start]
        samecolor = 1
        space = 0
        start1 = start
        while True:
            start += int(self.NS +1)
            if self.board[start] == color:
                samecolor += 1
            else:
                if self.board[start] == EMPTY:
                    space += 1
                break
        while True:
            start1 -= int(self.NS + 1)
            if self.board[start1] == color:
                samecolor += 1
            else:
                if self.board[start1] == EMPTY:
                    space += 1
                break
        return samecolor, space
    def checkd2(self, start,color):
        checkColor = self.board[start]
        samecolor = 1
        space = 0
        start1 = start
        while True:
            start += int(self.NS -1)
            if self.board[start] == color:
                samecolor += 1
            else:
                if self.board[start] == EMPTY:
                    space += 1
                break
        while True:
            start1 -= int(self.NS - 1)
            if self.board[start1] == color:
                samecolor += 1
            else:
                if self.board[start1] == EMPTY:
                    space += 1
                break
        return samecolor, space
    def checkrow2(self, start,color):
        checkColor = self.board[start]
        samecolor = 1
        space = 0
        start1 = start
        d = 0
        s1 = -1
        s2 = -1
        while True:
            start += 1
            if self.board[start] == color:
                samecolor += 1
                d = 1
            else:
                if self.board[start] == EMPTY:
                    space += 1
                    s1 = start
                break
        while True:
            start1 -= 1
            if self.board[start1] == color:
                if d == 1:
                    d = 0
                else:
                    d = -1
                samecolor += 1
            else:
                if self.board[start1] == EMPTY:
                    space += 1
                    s2 = start1
                break
        return samecolor, space, d, s1, s2
    def checkcol2(self, start,color):
        checkColor = self.board[start]
        samecolor = 1
        space = 0
        start1 = start
        d = 0
        s1 = -1
        s2 = -1
        while True:
            start += self.NS
            if self.board[start] == color:
                d = 1
                samecolor += 1
            else:
                if self.board[start] == EMPTY:
                    space += 1
                    s1 = start
                break
        while True:
            start1 -= self.NS
            if self.board[start1] == color:
                if d == 1:
                    d = 0
                else:
                    d = -1
                samecolor += 1
            else:
                if self.board[start1] == EMPTY:
                    space += 1
                    s2 = start1
                break
        return samecolor, space, d, s1, s2
    def checkd12(self, start,color):
        checkColor = self.board[start]
        samecolor = 1
        space = 0
        start1 = start
        d = 0
        s1 = -1
        s2 = -1
        while True:
            start += int(self.NS +1)
            if self.board[start] == color:
                samecolor += 1
                d = 1
            else:
                if self.board[start] == EMPTY:
                    space += 1
                    s1 = start
                break
        while True:
            start1 -= int(self.NS + 1)
            if self.board[start1] == color:
                if d == 1:
                    d = 0
                else:
                    d = -1
                samecolor += 1
            else:
                if self.board[start1] == EMPTY:
                    space += 1
                    s2 = start1
                break
        return samecolor, space, d, s1, s2
    def checkd22(self, start,color):
        checkColor = self.board[start]
        samecolor = 1
        space = 0
        start1 = start
        d = 0
        s1 = -1
        s2 = -1
        while True:
            start += int(self.NS -1)
            if self.board[start] == color:
                samecolor += 1
                d = 1
            else:
                if self.board[start] == EMPTY:
                    space += 1
                    s1 = start
                break
        while True:
            start1 -= int(self.NS - 1)
            if self.board[start1] == color:
                if d == 1:
                    d = 0
                else:
                    d = -1
                samecolor += 1
            else:
                if self.board[start1] == EMPTY:
                    space += 1
                    s2 = start1
                break
        return samecolor, space, d, s1, s2
    def Win(self):
        move_list = []
        color = self.current_player
        moves = where1d(self.board == EMPTY)
        for i in moves:
            rowc,rows = self.checkrow(i,color)
            colc,cols = self.checkcol(i,color)
            d1c,d1s = self.checkd1(i,color)
            d2c,d2s = self.checkd2(i,color)
            if rowc>=5 or colc >= 5 or d1c >= 5 or d2c >= 5:
                move_list.append(i)
        return move_list
    def BlockWin(self):
        color = WHITE + BLACK - self.current_player
        moves = where1d(self.board == EMPTY)
        move_list = []
        for i in moves:
            rowc,rows = self.checkrow(i,color)
            colc,cols = self.checkcol(i,color)
            d1c,d1s = self.checkd1(i,color)
            d2c,d2s = self.checkd2(i,color)
            if rowc>=5 or colc >= 5 or d1c >= 5 or d2c >= 5:
                move_list.append(i)
        return move_list
    def OpenFour(self):
        color = self.current_player
        moves = where1d(self.board == EMPTY)
        move_list = []
        for i in moves:
            rowc,rows = self.checkrow(i,color)
            colc,cols = self.checkcol(i,color)
            d1c,d1s = self.checkd1(i,color)
            d2c,d2s = self.checkd2(i,color)
            if rowc ==4 and rows == 2:
                move_list.append(i)
            elif colc ==4 and cols == 2:
                move_list.append(i)
            elif d1c ==4 and d1s == 2:
                move_list.append(i)
            elif d2c ==4 and d2s ==2:
                move_list.append(i)
        return move_list
    def BlockOpenFour(self):
        color = WHITE + BLACK - self.current_player
        moves = where1d(self.board == EMPTY)
        move_list = []
        for i in moves:
            rowc,rows,rd, rs1,rs2 = self.checkrow2(i,color)
            colc,cols,cd, cs1, cs2= self.checkcol2(i,color)
            d1c,d1s,d1, ds1, ds2 = self.checkd12(i,color)
            d2c,d2s,d2, d21, d22 = self.checkd22(i,color)


            if rowc ==4 and rows == 1:
                if rd == 1 and self.board[i+5] == 0:
                    move_list.append(i+5)
                    move_list.append(i)
                elif rd == -1 and self.board[i-5] == 0:
                    move_list.append(i+5)
                    move_list.append(i)
            if colc ==4 and cols == 1:
                if cd == 1:
                    if self.board[i+5*self.NS] == 0:
                        move_list.append(i+5*self.NS)
                        move_list.append(i)
                elif cd == -1:
                    if self.board[i-5*self.NS] == 0:
                        move_list.append(i-5*self.NS)
                        move_list.append(i)
            if d1c ==4 and d1s == 1:
                if d1 == 1:
                    if self.board[i+5*(self.NS+1)] == 0:
                        move_list.append(i+5*(self.NS+1))
                        move_list.append(i)
                elif d1 == -1:
                    if self.board[i-5*(self.NS+1)] == 0:
                        move_list.append(i-5*(self.NS+1))
                        move_list.append(i)
            if d2c ==4 and d2s ==1:
                if d2 == 1:
                    if self.board[i+5*(self.NS-1)] == 0:
                        move_list.append(i+5*(self.NS-1))
                        move_list.append(i)
                elif d2 == -1:
                    if self.board[i-5*(self.NS-1)] == 0:
                        move_list.append(i-5*(self.NS-1))
                        move_list.append(i)

            if rowc ==4 and rows == 2:
                move_list.append(i)
            if colc ==4 and cols == 2:
                move_list.append(i)
            if d1c ==4 and d1s == 2:
                move_list.append(i)
            if d2c ==4 and d2s ==2:
                move_list.append(i)

            if rowc ==4 and rows == 2 and rd == 0:
                move_list.append(rs1)
                move_list.append(rs2)
            if colc ==4 and cols == 2 and cd == 0:
                move_list.append(cs1)
                move_list.append(cs2)
            if d1c ==4 and d1s == 2 and d1 == 0:
                move_list.append(ds1)
                move_list.append(ds2)
            if d2c ==4 and d2s ==2 and d2 == 0:
                move_list.append(d21)
                move_list.append(d22)

        move_list = list(set(move_list))

        return move_list
