# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 22:22:13 2019
[move,wins,playouts,[childnodes]]
@author: 44775
"""

#reading from linked list.

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 19:04:00 2019

@author: 44775
"""

""".alphaBeta algorithm with an opening book.
 In order to load the book only once, enters a game play loop with Y/N Q asked if you want to continue playing.
alphaBeta algorithm with an opening book."""

import os
import pickle
import chess
import random
import math

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
        
    def Update(self,value,multiplier):
        self.playouts += 1*multiplier
        self.wins += value*multiplier
        
    def SelectChildPureMC(self):
        """ Selects the move based on the UCT formula. """        
        epsilon = float(1e-6)
        selectedMove = sorted(self.childNodes, key = lambda c: c.wins/(c.playouts + epsilon) + math.sqrt(2*math.log(self.playouts)/(c.playouts+epsilon)))[-1]
        
        return selectedMove
    
    def AddChild(self, m, board):
        """
        Remove the move from vUntriedMoves and add a new child node 
        """
        newNode = Node(move = m, parent = self, board = board)
        self.vUntriedMoves.remove(m)
        self.childNodes.append(newNode)
        return newNode

def NodeConverter(Node):
    """Converts a node into a linked list equivalent for saving and storage purposes."""  
    return [Node.move,Node.wins,Node.playouts,[NodeConverter(cnode) for cnode in Node.childNodes]]

def MhattDist(a,b):
    """ Returns the manhattan distance between two squares """
    filedist = abs((a % 8)-(b % 8))
    rankdist = abs(math.floor(a/8) - math.floor(b/8))

    return rankdist + filedist

def ENDGAMEFunc(board):
    """ Evaluates strength of a position based on endgame specific metrics """
    if board.is_checkmate():
        if board.turn:
            value = -2000
        else:
            value = 2000
    else:
        """ Evaluates a position based on Endgame parameters, pushed pawns and King activation """
        value = len(board.pieces(1,True)) + 3*len(board.pieces(2,True)) + 3*len(board.pieces(3,True)) + 5*len(board.pieces(4,True)) + 9*len(board.pieces(5,True)) + 150*len(board.pieces(6,True))
        value = value - len(board.pieces(1,False)) - 3*len(board.pieces(2,False)) - 3*len(board.pieces(3,False)) - 5*len(board.pieces(4,False)) - 9*len(board.pieces(5,False)) - 150*len(board.pieces(6,False))
    
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
        value = len(board.pieces(1,True)) + 3.25*len(board.pieces(2,True)) + 3.25*len(board.pieces(3,True)) + 5*len(board.pieces(4,True)) + 9*len(board.pieces(5,True)) + 150*len(board.pieces(6,True))
        value = value - len(board.pieces(1,False)) - 3.25*len(board.pieces(2,False)) - 3.25*len(board.pieces(3,False)) - 5*len(board.pieces(4,False)) - 9*len(board.pieces(5,False)) - 150*len(board.pieces(6,False))
    
        """KING POSITIONAL BONUS"""
        """ RANK = floor(kingpos/8), FILE = KingPos % 8"""
        kingposW = board.king(True)
        value = value + 0.2*KingMoveTable[math.floor(kingposW/8)][kingposW % 8]
        
        kingposB = board.king(False)
        value = value + 0.2*KingMoveTable[math.floor(kingposB/8)][kingposB % 8]
        
        """ Rook Positional Bonus"""
        rooksW = [square for square in board.pieces(4,True)]
        for rook in rooksW:
            value = value + 0.2*RookMoveTableW[math.floor(rook/8)][rook % 8]
            
        rooksB = [square for square in board.pieces(4,False)]
        for rook in rooksB:
            value = value + 0.2*RookMoveTableB[math.floor(rook/8)][rook % 8]
        
        """ Pawn Positional Bonus"""
        pawnsW = [square for square in board.pieces(4,True)]
        for pawn in pawnsW:
            value = value + 0.2*PawnMoveTableW[math.floor(pawn/8)][pawn % 8]
            
        pawnsB = [square for square in board.pieces(4,False)]
        for pawn in pawnsB:
            value = value + 0.2*PawnMoveTableB[math.floor(pawn/8)][pawn % 8]
            
        """ Bishop Pair Bonus """
        if (len(board.pieces(3,True)) == 2):
            value = value+0.25        
        if (len(board.pieces(3,False)) == 2):
            value = value-0.25
            
    return value


def BoardEval(board):
     """ Hash the board """
     """ Takes a blend of endgame and midgame evaluation funcs in attempt to prevent eval discontinuity."""
     phase = (len(board.pieces(5,True))*8+len(board.pieces(5,False)))*8+(len(board.pieces(3,True))*3+len(board.pieces(3,False)))*3 + (len(board.pieces(2,True))*3+len(board.pieces(2,False)))*3+(len(board.pieces(4,True))*5+len(board.pieces(4,False)))*5 + (len(board.pieces(1,True))+len(board.pieces(1,False)))
     val = (phase/68)*MIDGAMEFunc(board) + (68-phase)/68*ENDGAMEFunc(board)
     
     return [val]
 
def calcMinimaxMoveBook(board,depth,isMaximizingPlayer,alpha,beta):
    
    if (depth == 0) or board.is_game_over(claim_draw=False):
        return BoardEval(board)
    
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
            newboard.push_uci(validMoves[index].uci())
            moveval = calcMinimaxMoveBook(newboard,depth-1,not(isMaximizingPlayer),alpha,beta)[0]
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

"""Match between Minimax and Minimax with book:"""

""" Iterative Deepening with transposition table gainst human player """
def HumanMachineMatch(depth):
    
    Cnode = OpeningBook        
    alpha = float("-inf")
    beta = float("inf")
    depth = 3
    board = chess.Board()
    moveclock = 0
    
    """ Keeps track of the node."""
    criticalLvl = 400
    while (not board.is_game_over(claim_draw=False)):
        moveclock+=1
        if (board.turn):
            if moveclock < 8 and Cnode[2] > criticalLvl:
                """Look in the book"""
                print('In Book')
                if moveclock == 1:
                    """Pick one of the top two moves"""
                    indx = random.randrange(-2, 0)                        
                else:
                    """Larger range means more variability in play."""
                    indx = random.randrange(-1, 0)
                
                children = [x for x in Cnode[3]]
#                for node in children:
#                    print(node[0])
#                    print(node[1])
#                    print(node[2])
#                    print('----')
                    
                bestChild = sorted(children, key = lambda c: c[1]/c[2])[indx]
                Cnode = bestChild
                board.push_uci(Cnode[0].uci())
                print(Cnode[0])
            else:
                """Run AB Minimax search as standard."""
                smove = calcMinimaxMoveBook(board,depth,board.turn,alpha,beta)                
                board.push_uci(smove[1].uci())
                #display(board)
                print(smove[1])
        else:
            movestr = input("Make your move in SAN.")
            board.push_san(movestr)
#            display(board)
            if moveclock < 8:
                for node in Cnode[3]:
                    if (node[0] == board.peek()):
                        Cnode = node
                        break

def WhiteBookSim(d1,d2):
    """Simulates a match between two alpha beta algos, but white has acess to an opening book."""
    Cnode = OpeningBook
    alpha = float("-inf")
    beta = float("inf")
    board = chess.Board()
    moveclock = 0
    criticalLvl = 400
    while (not board.is_game_over(claim_draw=False)):
        moveclock+=1
        if (board.turn):
            if moveclock < 8 and Cnode[2] > criticalLvl:
                """Look in the book"""
                if moveclock == 1:
                    """Pick one of the top two moves"""
                    indx = random.randrange(-2, 0)                        
                else:
                    """Larger range means more variability in play."""
                    indx = random.randrange(-1, 0)
                children = [x for x in Cnode[3]]
                bestChild = sorted(children, key = lambda c: c[1]/c[2])[indx]
                Cnode = bestChild
                board.push_uci(Cnode[0].uci())
                print(Cnode[0])
            else:
                """Run AB Minimax search as standard."""
                smove = calcMinimaxMoveBook(board,d1,board.turn,alpha,beta)                
                board.push_uci(smove[1].uci())
        else:
            smove = calcMinimaxMoveBook(board,d2,board.turn,alpha,beta)              
            board.push_uci(smove[1].uci())
            if moveclock < 8:
                for node in Cnode[3]:
                    if (node[0] == board.peek()):
                        Cnode = node
                        break
    if (board.result() == '1/2-1/2'):
        d = 1
    elif(board.result() == '1-0'):
        w = 1
    else:
        l = 1
    return w,d,l

def BlackBookSim(d1,d2):
    """Simulates a match between two alpha beta algos, but white has acess to an opening book."""
    Cnode = OpeningBook
    alpha = float("-inf")
    beta = float("inf")
    board = chess.Board()
    moveclock = 0
    criticalLvl = 400
    while (not board.is_game_over(claim_draw=False)):
        moveclock+=1
        if (board.turn):
            smove = calcMinimaxMoveBook(board,d1,board.turn,alpha,beta)              
            board.push_uci(smove[1].uci())
            if moveclock < 8:
                for node in Cnode[3]:
                    if (node[0] == board.peek()):
                        Cnode = node
                        break
        else:
            if moveclock < 8 and Cnode[2] > criticalLvl:
                """Look in the book"""
                if moveclock == 1:
                    """Pick one of the top two moves"""
                    indx = random.randrange(-2, 0)                        
                else:
                    """Larger range means more variability in play."""
                    indx = random.randrange(-1, 0)
                children = [x for x in Cnode[3]]
                bestChild = sorted(children, key = lambda c: c[1]/c[2])[indx]
                Cnode = bestChild
                board.push_uci(Cnode[0].uci())
                print(Cnode[0])
            else:
                """Run AB Minimax search as standard."""
                smove = calcMinimaxMoveBook(board,d2,board.turn,alpha,beta)                
                board.push_uci(smove[1].uci())
    if (board.result() == '1/2-1/2'):
        d = 1
    elif(board.result() == '1-0'):
        w = 1
    else:
        l = 1
    return w,d,l
    
def CompareBookVanilla(d1,d2,Q):
    """Plays equal number of games as white and black with the Minimax algorithms, one has access to an opening book."""
    w=0
    d=0
    l=0
    for i in range(math.ceil(Q/2)):
        nw, nd, nl = WhiteBookSim(d1,d2)
        w +=nw 
        d +=nd
        l += nl
    
    for i in range(math.ceil(Q/2),Q):
        nl, nd, nw = BlackBookSim(d2,d1)
        w +=nw 
        d +=nd
        l += nl
        
    print('Win % ' + str(w/len(range(Q))))
    print('Draw % '+ str(d/len(range(Q))))
    print('Loss % '+ str(l/len(range(Q))))
    return 'Finished'
    


FileDir = os.path.dirname(os.path.abspath(__file__))
filename = FileDir + "/LinkedBk1.txt"

file_OB = open(filename,'rb')
OpeningBook = pickle.load(file_OB)

#HumanMachineMatch(4)

CompareBookVanilla(2,2,5)