# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:35:30 2019

@author: 44775
"""
import numpy as np
import random
import chess

""" Genetic Algorithm Experiment"""

def BoardEval(board,weights):
    """ Takes in a board and evaluates it based on a function that changes based on the weights fed"""
    value = weights[0]*float(len(board.pieces(1,True))) + weights[1]*float(len(board.pieces(2,True))) + weights[2]*float(len(board.pieces(3,True))) + weights[3]*float(len(board.pieces(4,True))) + weights[4]*float(len(board.pieces(5,True)))
    value = value - weights[0]*float(len(board.pieces(1,False))) - weights[1]*float(len(board.pieces(2,False))) - weights[2]*float(len(board.pieces(3,False))) - weights[3]*float(len(board.pieces(4,False))) - weights[4]*float(len(board.pieces(5,False)))
    return [value]

def calcMinimaxMoveWeighted(board,depth,isMaximizingPlayer,alpha,beta,Weights):
    
    if (depth == 0) or board.is_game_over(claim_draw=False):
        return BoardEval(board,Weights) 
    
    """ search for best possible move """
    bestmove = []
    
    if (isMaximizingPlayer):
        bestmovevalue = alpha
    else:
        bestmovevalue = beta
        
    validMoves = [move for move in board.legal_moves]
    """ Sorts moves to have Captures first to improve alphabeta pruning efficiency."""
    random.shuffle(validMoves)
    validMoves.sort(key=board.is_capture)
    
    " If only one possibility"""
    if (len(validMoves) == 1):
        newboard = board.copy()
        newboard.push_uci(validMoves[0].uci())
        bestmove = validMoves[0]
        bestmovevalue = BoardEval(newboard,Weights)[0]
    else:
        for index in range(len(validMoves)):
            """ Make the move run function on child, update values, then undo the move."""
            newboard = board.copy()
            newboard.push_uci(validMoves[index].uci())
            moveval = calcMinimaxMoveWeighted(newboard,depth-1,not(isMaximizingPlayer),alpha,beta,Weights)[0]
            #display(moveval)
            
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


def AlgoDuel(weightsA,weightsB,Quantity):
    " Takes in a set of two pairs of weights, and battles two Minimax algos with those weigths against each other X times, records the results."""
    alpha = float('-inf')
    beta  = float('inf')
    depth = 2
    results = []
    for num in range(0,Quantity):
        board = chess.Board()
        while (not board.is_game_over(claim_draw=False)):
                if (board.turn):
                    board.push_uci(calcMinimaxMoveWeighted(board,depth,board.turn,alpha,beta,weightsA)[1].uci())
                else:
                    board.push_uci(calcMinimaxMoveWeighted(board,depth,board.turn,alpha,beta,weightsB)[1].uci())
        results.append(board.result())
    return [results.count('1-0'),results.count('0-1')]

def RunBasicGeneticExperiement(InitialWeight,Generations,Playouts):
    """ Generates random array of weights """
    WeightA = InitialWeight
    counter = 0
    while counter<Generations:
        WeightB = np.random.uniform(0,4,(1,5)).tolist()[0]
        [W,L] = AlgoDuel(WeightA,WeightB,Playouts)
        Tot = W+L
        if (Tot!=0):
            WeightA = (W/Tot)*np.asarray(WeightA) + (L/Tot)*np.asarray(WeightA)
        else:
            WeightA = 0.5*WeightA + 0.5*WeightB
        counter +=1
        print("Generation: = " + str(counter) + " Weights:= " + str(WeightA))
    return WeightA

def RunAdvancedGeneticExperiement(InitialWeight,Generations,Playouts):
    """ Ep stands for epsilon that represents the quantity of intergenerational mutation, it decays slowly over time. 
    MaskList is a list of masks used for constructing the 8 children each generation. """
    Ep = 0.5 
    
    WeightA = InitialWeight
    bestmaskval = 0
    counter = 0
    
    MaskList = [[Ep,0,0,0,0],
                [0,Ep,0,0,0],
                [0,0,Ep,0,0],
                [0,0,0,Ep,0],
                [0,0,0,0,Ep],
                [-Ep,0,0,0,0],
                [0,-Ep,0,0,0],
                [0,0,-Ep,0,0],
                [0,0,0,-Ep,0],
                [0,0,0,0,-Ep]]
    
    while counter<Generations:
        """ Each generation dad and his 8 kids duke it out, keeps track of the strongest boi"""
        innerCount = 0
        print("dueling ")
        for mask in MaskList:
            innerCount+=1
            print(str(innerCount) + "/10")
            WeightB = [x+y for x,y in zip(WeightA,mask)]
            """ Duels the two AI and records results"""
            [W,L] = AlgoDuel(WeightA,WeightB,Playouts)
            
            if (W+L==0):
                Tot = 2
            else:
                Tot = W+L
            if (L/Tot > bestmaskval):
                """ More losses => better to switch."""
                bestmask = mask
                bestmaskval = L/Tot
        """ Generates the next Generation from the best mask from prev gen."""
        """ It shifts more depending on number of wins and losses."""
        if L > 0:
            print("Losses/Wins+Losses : " + str(bestmaskval))
            print("Best weights: " + str([x+y for x,y in zip(WeightA,bestmask)]))
            WeightA = (bestmaskval)*np.asarray([x+y for x,y in zip(WeightA,bestmask)]) + (1-bestmaskval)*np.asarray(WeightA)
            """ Epsilon reduces slightly, this is equiv to cooling in simulated Annealing,
            or think of this as reducing diet of radioactive chems in a GA. """
            Ep *= 0.95
            counter +=1
        else:
            print("Initial Weights still best")
            
        print("Generation: = " + str(counter) + " Current Weights:= " + str(WeightA))
    return WeightA

board = chess.Board()  
"""
RunBasicGeneticExperiement(np.array([1,1,1,1,2]),5,3)
"""
RunAdvancedGeneticExperiement([1,1,1,1,1],15,5)
        
        
        
        
    
    