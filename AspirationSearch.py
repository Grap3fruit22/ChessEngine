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

from ChessCore import BoardEval

def calcMinimaxMoveLMTT(board,depth,isMaximizingPlayer,alpha,beta):
    if (depth == 0) or board.is_game_over(claim_draw=False):
        val = BoardEval(board)
        return val    
    """ search for best possible move """
    bestmove = []
    if (isMaximizingPlayer):
        bestmovevalue = float("-inf")
    else:
        bestmovevalue = float("inf")
       
    validMoves = [move for move in board.legal_moves]
    """ Sorts moves to have Captures first to improve alphabeta pruning efficiency."""
    random.shuffle(validMoves)
    validMoves.sort(key=board.is_capture)
    
    " If only one possibility"""
    if (len(validMoves) == 1):
        newboard = board.copy()
        newboard.push_uci(validMoves[0].uci())
        bestmove = validMoves[0]
        bestmovevalue = BoardEval(newboard)[0]
    else:
        for index in range(len(validMoves)):
            """ Make the move run function on child, update values, then undo the move."""
            newboard = board.copy()
            if (not(board.is_capture(validMoves[index])) and depth > 1):
                LMvalid = True
            else:
                LMvalid = False
                
            newboard.push_uci(validMoves[index].uci())
            if (LMvalid and BoardEval(board)[0] < alpha):
                """ Late move reduction by 1 less ply """
                moveval = calcMinimaxMoveLMTT(newboard,depth-2,isMaximizingPlayer,alpha,beta)[0]
            else:    
                moveval = calcMinimaxMoveTT(newboard,depth-1,not(isMaximizingPlayer),alpha,beta)[0]
                
            if (isMaximizingPlayer):
                """ Attempt to maximize the position """
                if (moveval > bestmovevalue): 
                    bestmove = validMoves[index]
                    bestmovevalue = moveval
                alpha = max(alpha,moveval)
                
                if (beta <= alpha):
                    break
                return [alpha,bestmove]
            else:
                """ Attempt to minimize the position """
                if (moveval < bestmovevalue):
                    bestmove = validMoves[index]
                    bestmovevalue = moveval
                beta = min(beta,moveval)
                
                if (beta <= alpha):
                    break
                return [beta,bestmove]
    return [bestmovevalue,bestmove]

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
