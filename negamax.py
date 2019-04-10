# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:14:23 2019

@author: 44775
"""
import chess
import random

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
     
     return val
 
def negaMax(board, alpha, beta, depth, pv):
    
    line = []
    
    if depth == 0:
        return BoardEval(board), line
    
    for move in board.legal_moves:
        board.push_uci(move.uci())
        score, childPV = negaMax(board, -beta, -alpha, depth-1,pv)
        score *=-1
        board.pop()
        if score >= beta:
            pv.clear
            pv = [move] + childPV
            return beta, pv
        if score > alpha:
            alpha = score
            pv.clear
            #del pv[-1:-(depth+1)]
            pv = [move] + childPV
            
    return alpha, pv
        

def HumanMachineMatch(depth):
    alpha = float("-inf")
    beta = float("inf")
    depth = 3
    board = chess.Board()
    moveclock = 0
    """ Keeps track of the node."""
    #Cnode = OpeningBook
    criticalLvl = 100
    while (not board.is_game_over(claim_draw=False)):
        moveclock+=1
        if (board.turn):
            if moveclock < 0: #and Cnode.playouts > criticalLvl:
                """Look in the book"""
                display('In Book')
                if moveclock == 1:
                    """Pick one of the top three moves"""
                    indx = random.randrange(-2, 0)                        
                else:
                    """Pick one of the top two moves"""
                    indx = random.randrange(-2, 0)
                for node in (sorted(Cnode.childNodes, key = lambda c: c.wins/c.playouts)):
                    print(node.move)
                    print(node.wins)
                    print(node.playouts)
                    print('----')
                Cnode = sorted(Cnode.childNodes, key = lambda c: c.wins/c.playouts)[indx]
                board.push_uci(Cnode.move.uci())
                display(board)
            else:
                """Run AB Minimax search as standard."""
                valuation, PV = negaMax(board,alpha,beta,depth,[])
                print(PV)
                board.push_uci(PV[0].uci())
                display(board)
        else:
            movestr = input("Make your move in SAN.")
            board.push_san(movestr)
            display(board)
            if moveclock < 0:
                for node in Cnode.childNodes:
                    if (node.move == board.peek()):
                        Cnode = node
                        break
HumanMachineMatch(4)