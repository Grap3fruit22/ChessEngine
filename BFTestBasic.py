# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 17:38:15 2019

@author: 44775
"""

import chess
import chess.syzygy
import chess.polyglot

#import pdb; pdb.set_trace()
global totalNodes

totalNodes = 0

from ChessCore import calcMinimaxMoveBF

def CalcBranchFact(Quantity):
    """Calculates the average branching fact per move, on average
    for a game, and then takes this avg over X number of games."""

    alpha = float("-inf")
    beta = float("inf")
    depth = 1
    depthmax = 4
    arr = [0] * 781
    TT = {}
    GameBF = []
    EpochBF = []
    
    for num in range(0,Quantity):
        """Loops through to play X games."""
        board = chess.Board()
        totalNodes = 0
        MoveBF = []
        GameBF = []
        while (not board.is_game_over(claim_draw=False)):
            depth = 1
            totalNodes = 0

            if (board.turn):
                """Plays for W"""
                moveval, move, totalNodes, null = calcMinimaxMoveBF(board,depth,board.turn,alpha,beta,0,[])
                depth += 1
                
                while(depth<depthmax):
                    moveval, move, totalNodes, MoveBF = calcMinimaxMoveBF(board,depth,board.turn,alpha,beta,0,[])
                    display([moveval, move])
                    depth += 1
            
                GameBF.append(sum(MoveBF)/len(MoveBF))    
                board.push_uci(move.uci())
            else:
                """Plays for B"""
                moveval, move, totalNodes, null = calcMinimaxMoveBF(board,depth,board.turn,alpha,beta,0,[])
                depth += 1
                
                while(depth<depthmax):
                    moveval, move, totalNodes, MoveBF = calcMinimaxMoveBF(board,depth,board.turn,alpha,beta,0,[])
                    depth += 1
            
                GameBF.append(sum(MoveBF)/len(MoveBF))
                board.push_uci(move.uci())
        """Keeps track of each game BF as its recorded"""
        EpochBF.append(sum(GameBF)/len(GameBF))
        
    return EpochBF, sum(EpochBF)/Quantity


X, Y = CalcBranchFact(1)
print(X)
print(Y)