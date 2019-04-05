# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:16:15 2019

@author: 44775
"""

""" EPT Algorithm """
import chess
import math
import random

def ETPValFunc(board):
    """ Valuation function that takes a board position and outputs the value of position.
    Is slimmed down compared to the Minimax valuation function, because Lorentz suggested
    faster less domain knowledge dependent valuation function performed better.
    This makes intuitive sense because the valuation function is triggered far more, due to
    monte carlo tree search sims."""
    phase = (len(board.pieces(5,True))*8+len(board.pieces(5,False)))*8+(len(board.pieces(3,True))*3+len(board.pieces(3,False)))*3 + (len(board.pieces(2,True))*3+len(board.pieces(2,False)))*3+(len(board.pieces(4,True))*5+len(board.pieces(4,False)))*5 + (len(board.pieces(1,True))+len(board.pieces(1,False)))
    val = (phase/68)*SimpleMidgameFunc(board) + (68-phase)/68*SimpleEndgameFunc(board)

    return val
 
def SimpleMidgameFunc(board):
    turn = board.turn
    
    if board.is_checkmate():
        if turn:
            value = -2000
        else:
            value = 2000

    else:
        """ Base Value Valuation """
        value = len(board.pieces(1,True)) + 3.25*len(board.pieces(2,True)) + 3.25*len(board.pieces(3,True)) + 5*len(board.pieces(4,True)) + 9*len(board.pieces(5,True))
        value = value - len(board.pieces(1,False)) - 3.25*len(board.pieces(2,False)) - 3.25*len(board.pieces(3,False)) - 5*len(board.pieces(4,False)) - 9*len(board.pieces(5,False))
    
        """ Bishop Pair Bonus """
        if (len(board.pieces(3,True)) == 2):
            value = value+0.25        
        if (len(board.pieces(3,False)) == 2):
            value = value-0.25
            
    return value

def MhattDist(a,b):
    """ Returns the manhattan distance between two squares """
    filedist = abs((a % 8)-(b % 8))
    rankdist = abs(math.floor(a/8) - math.floor(b/8))

    return rankdist + filedist

def SimpleEndgameFunc(board):
    """ Evaluates strength of a position based on endgame specific metrics """
    if board.is_checkmate():
        if board.turn:
            value = -2000
        else:
            value = 2000
    else:
        """ Evaluates a position based on Endgame parameters, pushed pawns and King activation """
        value = len(board.pieces(1,True)) + 3*len(board.pieces(2,True)) + 3*len(board.pieces(3,True)) + 5*len(board.pieces(4,True)) + 9*len(board.pieces(5,True))
        value = value - len(board.pieces(1,False)) - 3*len(board.pieces(2,False)) - 3*len(board.pieces(3,False)) - 5*len(board.pieces(4,False)) - 9*len(board.pieces(5,False))
    
        kingposW = board.king(True)
        kingposB = board.king(False)
        
        """ ACTIVATE the KING and PUSH PAWNS"""
        """ add value for past and connected past pawns """
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
        
        pawnsB = [square for square in board.pieces(1,False)]
        for pawn in pawnsB:
            """penalize manhattan dist from king"""
            value = value + 0.15*MhattDist(pawn,kingposB)
            pfilesB.add(pawn % 8)
            
            """Push em baby"""
            value = value - 0.05*(8 - (pawn - (pawn % 8))/8)
            
    return value

def ETPplayout(startboard,EarlyStop):
    """ Takes a board as an inputs,and a number of moves after which will terminate the simulation and apply the Valuation Function.
    and then plays a game from the board position by playing random moves until termination or game completion. """
    results = []
    board = chess.Board(startboard.fen())
    #display(board) Now fixed and is taking correct Board reset.
    MoveCount = 0
    while (not board.is_game_over(claim_draw=False)):
        
        rmove = random.choice([move for move in board.legal_moves])
        board.push_uci(rmove.uci())
        MoveCount+=1
        if MoveCount>EarlyStop:
            """Break out of Game"""
            break
        
    if MoveCount>EarlyStop:
        """Apply Val Func"""
        if board.turn:
            if ETPValFunc(board) > 0:
                val = 1
            else:
                val = 0
        else:
            if ETPValFunc(board) < 0:
                val = 1
            else:
                val = 0
    else:    
        results.append(board.result())
    
        if (board.result() == '1/2-1/2'):
            val = 0.5
        else:
            if (board.result() =='1-0'):
                val = 1
            else:
                """ !=draw & !=win => loss"""
                val = 0

    return val;

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
        selectedMove = sorted(self.childNodes, key = lambda c: c.wins/(c.playouts + epsilon) + math.sqrt(2*math.log(self.playouts)/(c.playouts+epsilon)))[-1]
        
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


    """EPT Adaptaion of MCTS"""
def ETP(rootfen,itermax,DepthCutout, verbose = False):
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
        """ Second argument is for EPT"""
        result = ETPplayout(chess.Board(board.fen()),DepthCutout)
        
        """BACKPROPOGATION"""
        while node != None:
            if node.col:
                node.Update(result)
                node = node.parentNode
            else:
                node.Update(1-result)
                node = node.parentNode
        
    return sorted(rootNode.childNodes, key = lambda c: c.playouts)[-1].move

def SimulateMatch(iterW,iterB,DepthCutoutA,DepthCutoutB):
    """ Takes input arguments of the maximum num of iterations for White and Black"""
    board = chess.Board()
    while (not board.is_game_over(claim_draw=False)):
        if (board.turn):
            board.push_uci(EPT(board.fen(),iterW,DepthCutoutA).uci())
            display(board)
        else:
            board.push_uci(EPT(board.fen(),iterB,DepthCutoutB).uci())
            display(board)
            
def HumanMachineMatch(iterW,DepthCutout):
    """ Takes input arguments of the maximum num of iterations for White and Black"""
    board = chess.Board()
    while (not board.is_game_over(claim_draw=False)):
        if (board.turn):
            board.push_uci(EPT(board.fen(),iterW,DepthCutout).uci())
            display(board)
        else:
            movestr = input("Make your move in SAN.")
            board.push_san(movestr)
            display(board)       
            
HumanMachineMatch(30000,2)





