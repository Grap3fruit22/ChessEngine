# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 17:38:15 2019

@author: 44775
"""

import chess
import chess.syzygy
import chess.polyglot
import matplotlib.pyplot as plt
import math

from ChessCore import calcMinimaxMoveBFADV, BoardEval


BucketEval = []

for i in range(0,30): BucketEval.append([])


def GetBranchFactStats(Quantity):
    """ Calculates a variety of metrics and stats for analysing branching factor
    in the game tree and how it is related to material, # moves deep, and the
    val function of the current position."""

    alpha = float("-inf")
    beta = float("inf")
    depth = 1
    depthmax = 3
    arr = [0] * 781
    TT = {}
    GameBF = []
    EpochBF = []
    
    for num in range(0,Quantity):
        """Loops through to play X games."""
        
        board = chess.Board()
        totalNodes = 0
        GameBF = []
        GameEval = []
        while (not board.is_game_over(claim_draw=False)):
            depth = 1
            totalNodes = 0
            MoveBF = []
            if (board.turn):
                """Plays for W"""
                moveval, move, totalNodes, null = calcMinimaxMoveBFADV(board,depth,board.turn,alpha,beta,0,[])
                depth += 1
                
                while(depth<depthmax):
                    moveval, move, totalNodes, MoveBF = calcMinimaxMoveBFADV(board,depth,board.turn,alpha,beta,0,[])
                    display([moveval, move])
                    depth += 1
            
                GameBF.append(sum(MoveBF)/len(MoveBF))
                GameEval.append(BoardEval(board)[0])
                board.push_uci(move.uci())
            else:
                """Plays for B"""
                moveval, move, totalNodes, null = calcMinimaxMoveBFADV(board,depth,board.turn,alpha,beta,0,[])
                depth += 1
                
                while(depth<depthmax):
                    moveval, move, totalNodes, MoveBF = calcMinimaxMoveBFADV(board,depth,board.turn,alpha,beta,0,[])
                    depth += 1
                
                GameBF.append(sum(MoveBF)/len(MoveBF))
                BucketEval[math.floor(abs(BoardEval(board)[0]))].append(sum(MoveBF)/len(MoveBF))
                GameEval.append(BoardEval(board)[0])
                board.push_uci(move.uci())
        """Keeps track of each game BF as its recorded"""
    return GameBF, GameEval, BucketEval #EpochBF, ##sum(EpochBF)/Quantity

X, Y, Z = GetBranchFactStats(2)
fig = plt.figure
print("Plot of Average branching factor for absolute eval func across X games")
plt.scatter(range(0,len(Z)), [sum(z)/float(len(z)) if len(z) !=0 else 0 for z in Z])

"""
fig = plt.figure
print("Valuation function through game")
plt.scatter(range(0,len(Y)), Y)
plt.xlabel('Game Move', fontsize = 16)
plt.ylabel('Absolute value of Board Valuation', fontsize = 16)
plt.show()

fig = plt.figure
print("----------------------")
plt.scatter(X,[abs(x) for x in Y])
plt.xlabel('Branching Factor', fontsize = 16)
plt.ylabel('Absolute value of Board Valuation', fontsize = 16)
plt.show()
"""
""" Output is a list of tuples containing (Branching Fact, Material, Move, Val Func)"""



