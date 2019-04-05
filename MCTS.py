# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 10:49:19 2019

@author: 44775
"""
import math
import random
import chess 
import chess.svg 

""" Single Playout Function"""
def playout(startboard):
    """ Takes a board as an inputs, and then plays a
    game from that position by playing random moves until game completion. """
    results = []
    for num in range(0,1):
        board = chess.Board(startboard.fen())
        #display(board) Now fixed and is taking correct Board reset.
        while (not board.is_game_over(claim_draw=False)):
            rmove = random.choice([move for move in board.legal_moves])
            board.push_uci(rmove.uci())
        results.append(board.result())
        
        if (results.count('1/2-1/2')==1):
            val = 0.5
        else:
            if (results.count('1-0')==1):
                val = 1
            else:
                val = 0
    return val;

def ValFunc2(node):
    """ Takes a position and evaluates it using a custom Val Function."""
    """Version 1 RAW Material:"""
    board = node.board
    turn = node.col
    
    val = len(board.pieces(1,not(turn))) + 3*len(board.pieces(2,not(turn))) + 3*len(board.pieces(3,not(turn))) + 5*len(board.pieces(4,not(turn))) + 9*len(board.pieces(5,not(turn))) + 150*len(board.pieces(6,not(turn)))
    val = val - len(board.pieces(1,turn)) - 3*len(board.pieces(2,turn)) - 3*len(board.pieces(3,turn)) - 5*len(board.pieces(4,turn)) - 9*len(board.pieces(5,turn)) - 150*len(board.pieces(6,turn))
    
    """ Bonus for Bishop Pair """
    if (len(board.pieces(3,turn)) == 2):
        val = val - 0.25
    if (len(board.pieces(3,not(turn))) == 2):
        val = val + 0.25
    
    return val
    
def ValFunc(node):
    """ Takes a position and evaluates it using a custom Val Function."""
    """Version 1 RAW Material:"""
    board = node.board
    turn = node.col
    
    val = len(board.pieces(1,not(turn))) + 3*len(board.pieces(2,not(turn))) + 3*len(board.pieces(3,not(turn))) + 5*len(board.pieces(4,not(turn))) + 9*len(board.pieces(5,not(turn))) + 150*len(board.pieces(6,not(turn)))
    val = val - len(board.pieces(1,turn)) - 3*len(board.pieces(2,turn)) - 3*len(board.pieces(3,turn)) - 5*len(board.pieces(4,turn)) - 9*len(board.pieces(5,turn)) - 150*len(board.pieces(6,turn))
    
    return val
    
""" Building the node class """
class Node:
    
    def __init__(self, move = None, parent = None, board = None):
        self.move = move
        self.board = board
        self.parentNode = parent
        self.childNodes = []
        self.wins = 0
        self.playouts = 0
        self.vUntriedMoves = [move for move in board.legal_moves]
        self.col = board.turn
        
    def Update(self,value):
        self.playouts += 1
        self.wins += value
        
    def SelectChildPureMC(self):
        """ Selects by using the UCT bounds. """
        epsilon = float(1e-6)
        selectedMove = sorted(self.childNodes, key = lambda c: c.wins/(c.playouts + epsilon) + math.sqrt(2*math.log(self.playouts)/(c.playouts+epsilon)))[-1]
        
        return selectedMove
        
    def SelectChildValFunc(self):
        """ Selects by using the Valuation function, Valfunc2 contains bonus for bishop pair. """
        selectedMove = sorted(self.childNodes, key = ValFunc2)[-1]
        
        return selectedMove    
    def AddChild(self, m, board):
        """
        Remove the move from vUntriedMoves and add a new child node 
        """
        newNode = Node(move = m, parent = self, board = board)
        self.vUntriedMoves.remove(m)
        self.childNodes.append(newNode)
        return newNode

    """MCTS"""
def PUREMCTS(rootfen,itermax, verbose = False):
    rootNode = Node(board = chess.Board(rootfen))
    
    for i in range(itermax):
        node = rootNode
        board = chess.Board(rootfen).copy()
        
        """SELECTION"""
        while (node.vUntriedMoves == []) and (node.childNodes != []):
            node = node.SelectChildPureMC()
            board.push_uci(node.move.uci())
        
        """EXPAND"""
        if node.vUntriedMoves != []:
            m = random.choice(node.vUntriedMoves)
            board.push_uci(m.uci())
            node = node.AddChild(m,chess.Board(board.fen()))
            
        """ SIMULATION"""
        result = playout(chess.Board(board.fen()))
        
        """BACKPROPOGATION"""
        while node != None:
            if node.col:
                node.Update(result)
                node = node.parentNode
            else:
                node.Update(1-result)
                node = node.parentNode
        
    return sorted(rootNode.childNodes, key = lambda c: c.playouts)[-1].move

def ADAPTMCTS(rootfen,itermax, verbose = False):
    rootNode = Node(board = chess.Board(rootfen))
    
    for i in range(itermax):
        node = rootNode
        board = chess.Board(rootfen).copy()
        
        """SELECTION"""
        while (node.vUntriedMoves == []) and (node.childNodes != []):
            node = node.SelectChildValFunc()
            board.push_uci(node.move.uci())
        
        """EXPAND"""
        if node.vUntriedMoves != []:
            m = random.choice(node.vUntriedMoves)
            board.push_uci(m.uci())
            node = node.AddChild(m,chess.Board(board.fen()))
            
        """ SIMULATION"""
        result = playout(chess.Board(board.fen()))
        
        """BACKPROPOGATION"""
        while node != None:
            if node.col:
                node.Update(result)
                node = node.parentNode
            else:
                node.Update(1-result)
                node = node.parentNode
        
    return sorted(rootNode.childNodes, key = lambda c: c.playouts)[-1].move 

"""Minimimax Evaluation Function"""
def MiniMaxValFunc(board):
    turn = board.turn
    
    """ Base Value Valuation """
    value = len(board.pieces(1,not(turn))) + 3*len(board.pieces(2,not(turn))) + 3*len(board.pieces(3,not(turn))) + 5*len(board.pieces(4,not(turn))) + 9*len(board.pieces(5,not(turn))) + 150*len(board.pieces(6,not(turn)))
    value = value - len(board.pieces(1,turn)) - 3*len(board.pieces(2,turn)) - 3*len(board.pieces(3,turn)) - 5*len(board.pieces(4,turn)) - 9*len(board.pieces(5,turn)) - 150*len(board.pieces(6,turn))
    
    """ Bishop Pair Bonus """
    if (len(board.pieces(3,not(turn))) == 2):
        value = value+0.25        
    if (len(board.pieces(3,turn)) == 2):
        value = value-0.25

    """ Past-Pawn Bonus"""
    enemypawns = [square for square in board.pieces(1,board.turn)]
    pawnfiles = set()
    for epawn in enemypawns:
        pawnfiles.add(epawn % 8)
    value = value + len(set(range(8))-pawnfiles)*0.5

    """ King Safety Bonus for defended squares near king """
    kingpos = [square for square in board.pieces(6,not(board.turn))][0]
    keysquares = []
    if kingpos % 8 == 0:
        """build corner struct"""
        if turn:
            keysquares.append(kingpos+1)
            keysquares.append(kingpos-7)
            keysquares.append(kingpos-8)
        else:
            keysquares.append(kingpos+1)
            keysquares.append(kingpos+8)
            keysquares.append(kingpos+9)
    else:
        if kingpos % 8 == 7:
            """Build other corner struct"""
            if turn:
                keysquares.append(kingpos-1)
                keysquares.append(kingpos-9)
                keysquares.append(kingpos-8)
            else:
                keysquares.append(kingpos-1)
                keysquares.append(kingpos+8)
                keysquares.append(kingpos+7)
        else:
            """Build wall struct"""
            if turn:
                keysquares.append(kingpos-9)
                keysquares.append(kingpos-8)
                keysquares.append(kingpos-7)
            else:
                keysquares.append(kingpos+9)
                keysquares.append(kingpos+8)
                keysquares.append(kingpos+7)
    
    """Counts attacks on the squares near the king, as a metric of king safety."""
    for square in keysquares:
        value = value + len(board.attackers(not(board.turn),square))*0.01
    
    return value

def calcMinimaxMove(rootfen,depth):
    """ List all Possible moves"""
    board = chess.Board(rootfen)
    validMoves = [move for move in board.legal_moves]
    """ Sort moves up randomly so not always picks lefthandside move first."""
    random.shuffle(validMoves)
    
    """ Exit if the game is over"""
    if (not board.is_game_over(claim_draw=False)):
        """ Search for highest Value move"""
        bestmove = []
        bestmoveValue = float("-inf")
        
        for index in range(len(validMoves)):
            """ Make the move evaluate board, update values, then undo the move."""
            newboard = board.copy()
            newboard.push_uci(validMoves[index].uci())
            moveval = MiniMaxValFunc(newboard)
            
            if (moveval > bestmoveValue): 
                bestmove = validMoves[index]
                bestmoveValue = moveval
            
    return bestmove

def SimulateMatch(iterW,iterB):
    """ Takes input arguments of the maximum num of iterations for White and Black"""
    board = chess.Board()
    while (not board.is_game_over(claim_draw=False)):
        if (board.turn):
            board.push_uci(ADAPTMCTS(board.fen(),iterW).uci())
            display(board)
        else:
            board.push_uci(PUREMCTS(board.fen(),iterB).uci())
            display(board)

def SimulateMatch2(iterW,iterB):
    """ Takes input arguments of the maximum num of iterations for White and Black"""
    board = chess.Board()
    while (not board.is_game_over(claim_draw=False)):
        if (board.turn):
            board.push_uci(ADAPTMCTS(board.fen(),iterW).uci())
            display(board)
        else:
            board.push_uci(calcMinimaxMove(board.fen(),iterB).uci())
            display(board)

SimulateMatch2(100,100)
""" Has significant issues, got mated within 4 moves on playing with a friend for testing, fell for the fools mate...
why is this: doesn't have enough simulations looking at short run of the game makes sense why it performs better in go with significant long run.

First attempt to deal with this is valuation func, and can also rather than do one simulations per node do two if not implement Minimax."""



