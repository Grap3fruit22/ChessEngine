# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:47:31 2019

@author: 44775
"""

""" Aspiration Search """


import chess
import chess.syzygy
import chess.polyglot
import random

from ChessCore import calcMinimaxMoveTT

alpha = float("-inf")
beta = float("inf")
depth = 1
depthmax = 5
""" 0.5 pawn wide Aspiration window. """
""" Trade off, more researching of the tree v.s faster searches """
Window = 0.45

arr = [0] * 781
TT = {}

board = chess.Board()

while (not board.is_game_over(claim_draw=False)):            
    if (board.turn):
        depth = 1
        """ Search depth 1 fully """
        smove = calcMinimaxMoveTT(board,depth,board.turn,alpha,beta)
        display(smove)
        depth += 1
        """ Prune aggresively by using a narrow Aspiration window for deeper searches"""
        while(depth<depthmax):
                alpha = smove[0] - Window
                beta = smove[0] + Window
                smove = calcMinimaxMoveTT(board,depth,board.turn,alpha,beta)
                """ if (smove[0] < alpha and smove[0] > beta):
                    print("Additional search triggered.")
                    smove = calcMinimaxMoveTT(board,depth,board.turn,float("-inf"),beta = float("inf"))"""
                
                display(smove)
                depth += 1
                
        board.push_uci(smove[1].uci())
        display(board)
    else:
        movestr = input("Make your move in SAN.")
        board.push_san(movestr)
        display(board)