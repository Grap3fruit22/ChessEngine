# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:37:58 2019

@author: 44775
"""
import chess
from ChessCore import calcMTDFMove

board = chess.Board()
depthmax = 5
depth = 1

while (not board.is_game_over(claim_draw=False)):            
    if (board.turn):
        depth = 1
        """ Search depth 1 fully """
        [bestmoveval, move]  = calcMTDFMove(board,depth,0)
        depth += 1
        """ Run MTD(f) Algorithm based on output of previous depth run"""
        while(depth<depthmax):
            [bestmoveval, move]  = calcMTDFMove(board,depth,bestmoveval)
            print([move,bestmoveval])
            depth += 1

        board.push_uci(move.uci())
        print('---')
        print(move)
        print('---')
    else:
        movestr = input("Make your move in SAN.")
        board.push_san(movestr)
        
        
