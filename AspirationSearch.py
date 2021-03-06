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

def calcMinimaxMoveTT(board,depth,isMaximizingPlayer,alpha,beta):
    if (depth == 0) or board.is_game_over(claim_draw=False):
        val = BoardEval(board)
        return val    
    """ search for best possible move """
    bestmove = []
    if (isMaximizingPlayer):
        bestmovevalue = alpha # float("-inf")
    else:
        bestmovevalue = beta # float("inf")
       
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
            newboard.push_uci(validMoves[index].uci()) 
            moveval = calcMinimaxMoveTT(newboard,depth-1,not(isMaximizingPlayer),alpha,beta)[0]
                
            if (isMaximizingPlayer):
                """ Attempt to maximize the position """
                if (moveval > bestmovevalue): 
                    bestmove = validMoves[index]
                    bestmovevalue = moveval
                alpha = max(alpha,moveval)
                
            else:
                """ Attempt to minimize the position """
                if (moveval < bestmovevalue):
                    bestmove = validMoves[index]
                    bestmovevalue = moveval
                beta = min(beta,moveval)
                
            if (beta <= alpha):
                break

    return [bestmovevalue,bestmove]

alpha = float("-inf")
beta = float("inf")
depth = 1
depthmax = 6
""" 0.5 pawn wide Aspiration window. """
""" Trade off, more researching of the tree v.s faster searches """
Window = [1.5,5,float("inf")]
windowCount = 0
arr = [0] * 781
TT = {}

board = chess.Board()

while (not board.is_game_over(claim_draw=False)):            
    if (board.turn):
        depth = 1
        """ Search depth 1 fully """
        smove = calcMinimaxMoveTT(board,depth,board.turn,alpha,beta)
        print(smove)
        depth += 1
        """ Prune aggresively by using a narrow Aspiration window for deeper searches"""
        while(depth<depthmax):
            windowCount = 0
            alpha = smove[0] - Window[windowCount] # float("-inf")
            beta = smove[0] + Window[windowCount] # float("inf")
            smove = calcMinimaxMoveTT(board,depth,board.turn,alpha,beta)
                
            """*** Exponential widening of the window. ***"""
            while smove[1] == []:
                windowCount +=1
                print("***Additional search triggered.***")
                alpha = alpha - Window[windowCount]
                beta = beta + Window[windowCount]
                smove = calcMinimaxMoveTT(board,depth,board.turn,alpha,beta)
                
            depth += 1
                
        board.push_uci(smove[1].uci())
        print(smove[1])
    else:
        movestr = input("Make your move in SAN.")
        board.push_san(movestr)
        print('---')
