# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:12:14 2019

@author: 44775
"""
from math import *
import random
import chess 


PawnMoveTableW = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.5, 1.0, 1.0, -2.0, -2.0, 1.0, 1.0, 0.5],
                  [0.5, -0.5, -1.0, 0.0, 0.0, -1.0, -0.5, 0.5],
                  [0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0],
                  [0.5, 0.5, 0.0, 2.5, 2.5, 0.0, 0.5, 0.5],
                  [1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0],
                  [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

PawnMoveTableB = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                  [1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0],
                  [0.5, 0.5, 0.0, 2.5, 2.5, 0.0, 0.5, 0.5],
                  [0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0],
                  [0.5, -0.5, -1.0, 0.0, 0.0, -1.0, -0.5, 0.5],
                  [0.5, 1.0, 1.0, -2.0, -2.0, 1.0, 1.0, 0.5],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

RookMoveTableB = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [-0.5, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.5],
                  [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
                  [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
                  [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
                  [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
                  [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
                  [0.0, 0.0, 0.0, -0.5, -0.5, 0.0, 0.0, 0.0]]

RookMoveTableW = [[0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0],
                  [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                  [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                  [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                  [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                  [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                  [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
                 
KingMoveTable = [[1.0, 2.5, 0.5, 0.0, 0.0, 0.5, 2.0, 2.0],
                 [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                 [-2.0, -3.5, -2.5, 0.0, 0.0, -0.5, -3.0, -2.0]]

def MhattDist(a,b):
    """ Returns the manhattan distance between two squares """
    filedist = abs((a % 8)-(b % 8))
    rankdist = abs(math.floor(a/8) - math.floor(b/8))

    return rankdist + filedist

def pieceCount(board,flag):
    val = len(board.pieces(flag,True))-len(board.pieces(flag,False))
    return val

def ENDGAMEFunc(board):
    """ Evaluates strength of a position based on endgame specific metrics """
    if board.is_checkmate():
        if board.turn:
            value = -2000
        else:
            value = 2000
    else:
        """ Evaluates a position based on Endgame parameters, pushed pawns and King activation """
        value = pieceCount(board,1) + 3.25*pieceCount(board,2) + 3.25*pieceCount(board,3) + 5*pieceCount(board,4) + 9*pieceCount(board,5)
    
        kingposW = board.king(True)
        kingposB = board.king(False)
        
        """ ACTIVATE the KING and PUSH PAWNS"""
        """ add value for past and connected past pawns, (and push em baby)"""
        pawnsW = [square for square in board.pieces(1,True)]
        pfilesW = set()
        pfilesB = set() 
        """ This is for adding past and connected pawn data."""
        
        for pawn in pawnsW:
            """penalize manhattan dist from king"""
            value = value - 0.15*MhattDist(pawn,kingposW)
            pfilesW.add(pawn % 8)
            
            """Push em baby"""
            value = value + 0.05*((pawn - (pawn % 8))/8)
            
            """Push em baby"""
            value = value + 0.05*(pawn - (pawn % 8))/8
        
        pawnsB = [square for square in board.pieces(1,False)]
        for pawn in pawnsB:
            """penalize manhattan dist from king"""
            value = value + 0.15*MhattDist(pawn,kingposB)
            pfilesB.add(pawn % 8)
            
            
            """Push em baby"""
            value = value - 0.05*(8 - (pawn - (pawn % 8))/8)
            
    return value
    
def MIDGAMEFunc(board):
    turn = board.turn
    
    if board.is_checkmate():
        if turn:
            value = -2000
        else:
            value = 2000

    else:
        """ Base Value Valuation """
        value = pieceCount(board,1) + 3.25*pieceCount(board,2) + 3.25*pieceCount(board,3) + 5*pieceCount(board,4) + 9*pieceCount(board,5)
    
        """KING POSITIONAL BONUS"""
        """ RANK = floor(kingpos/8), FILE = KingPos % 8"""
        kingposW = board.king(True)
        value = value + KingMoveTable[math.floor(kingposW/8)][kingposW % 8]
        
        kingposB = board.king(False)
        value = value + KingMoveTable[math.floor(kingposB/8)][kingposB % 8]
        
        """ Rook Positional Bonus"""
        rooksW = [square for square in board.pieces(4,True)]
        for rook in rooksW:
            value = value + 0.5*RookMoveTableW[math.floor(rook/8)][rook % 8]
            
        rooksB = [square for square in board.pieces(4,False)]
        for rook in rooksB:
            value = value + 0.5*RookMoveTableB[math.floor(rook/8)][rook % 8]
        
        """ Pawn Positional Bonus"""
        pawnsW = [square for square in board.pieces(4,True)]
        for pawn in pawnsW:
            value = value + 0.3*PawnMoveTableW[math.floor(pawn/8)][pawn % 8]
            
        pawnsB = [square for square in board.pieces(4,False)]
        for pawn in pawnsB:
            value = value + 0.3*PawnMoveTableB[math.floor(pawn/8)][pawn % 8]
            
        """ Bishop Pair Bonus """
        if (len(board.pieces(3,True)) == 2):
            value = value+0.25        
        if (len(board.pieces(3,False)) == 2):
            value = value-0.25
            
    return value

def MiniMaxValFunc(board):
     phase = (len(board.pieces(5,True))*8+len(board.pieces(5,False)))*8+(len(board.pieces(3,True))*3+len(board.pieces(3,False)))*3 + (len(board.pieces(2,True))*3+len(board.pieces(2,False)))*3+(len(board.pieces(4,True))*5+len(board.pieces(4,False)))*5 + (len(board.pieces(1,True))+len(board.pieces(1,False)))
     val = (phase/68)*MIDGAMEFunc(board) + (68-phase)/68*ENDGAMEFunc(board)
     return [val]

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
        """
        we want to pick out of the vUntriedMoves, based on a Valuation Function
        """
        """selectedMove = sorted(self.childNodes, key = ValuationFunc)[-1]"""
        
        epsilon = float(1e-6)
        selectedMove = sorted(self.childNodes, key = lambda c: c.wins/(c.playouts + epsilon) + sqrt(2*log(self.playouts)/(c.playouts+epsilon)))[-1]
        
        return selectedMove
        
    def SelectChildValFunc(self):
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

def calcMinimaxMove(board,depth,isMaximizingPlayer,alpha,beta):
    
    if (depth == 0) or board.is_game_over(claim_draw=False):
        return MiniMaxValFunc(board)
    
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
        bestmovevalue = MiniMaxValFunc(newboard)[0]
    else:
        for index in range(len(validMoves)):
            """ Make the move run function on child, update values, then undo the move."""
            newboard = board.copy()
            newboard.push_uci(validMoves[index].uci())
            moveval = calcMinimaxMove(newboard,depth-1,not(isMaximizingPlayer),alpha,beta)[0]
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
                
            """ Prune position """
            if (beta <= alpha):
                break
    
    return [bestmovevalue,bestmove]

def SimulateMatch2(depth,iterW):
    """ Takes input arguments of the maximum num of iterations for White and Black"""
    w = 0
    d = 0
    l = 0
    board = chess.Board()
    while (not board.is_game_over(claim_draw=False)):
        if (board.turn):
            board.push_uci(calcMinimaxMove(board,depth,board.turn,float(-inf),float(inf))[1].uci())
            #display(board)
        else:
            board.push_uci(ADAPTMCTS(board.fen(),iterW).uci())
            #display(board)
    
    if (board.result() == '1/2-1/2'):
        d = 1
    elif(board.result() == '1-0'):
        w = 1
    else:
        l = 1
    return w,d,l

def CompareAlgos(Q):
    w=0
    d=0
    l=0
    for i in range(Q):
        print(i/Q)
        nw, nd, nl = SimulateMatch2(3,50)
        w +=nw 
        d +=nd
        l += nl
        
    print('Win % ' + str(w/len(range(Q))))
    print('Draw % '+ str(d/len(range(Q))))
    print('Loss % '+ str(l/len(range(Q))))
    return 'Finished'


CompareAlgos(1)