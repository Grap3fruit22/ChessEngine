# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:12:14 2019

@author: 44775
"""
import math
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

def ETPValFunc(board):
    """ Takes a position and evaluates it using a custom Val Function."""
    """Version 1 RAW Material:"""
    turn = board.turn
    
    val = len(board.pieces(1,not(turn))) + 3*len(board.pieces(2,not(turn))) + 3*len(board.pieces(3,not(turn))) + 5*len(board.pieces(4,not(turn))) + 9*len(board.pieces(5,not(turn))) + 150*len(board.pieces(6,not(turn)))
    val = val - len(board.pieces(1,turn)) - 3*len(board.pieces(2,turn)) - 3*len(board.pieces(3,turn)) - 5*len(board.pieces(4,turn)) - 9*len(board.pieces(5,turn)) - 150*len(board.pieces(6,turn))
    
    """ Bonus for Bishop Pair """
    if (len(board.pieces(3,turn)) == 2):
        val = val - 0.2
    if (len(board.pieces(3,not(turn))) == 2):
        val = val + 0.2
    return val

def NodeValFunc(node):
    """ Takes a position and evaluates it using a custom Val Function."""
    """Version 1 RAW Material:"""
    board = node.board
    turn = node.col
    
    val = len(board.pieces(1,not(turn))) + 3*len(board.pieces(2,not(turn))) + 3*len(board.pieces(3,not(turn))) + 5*len(board.pieces(4,not(turn))) + 9*len(board.pieces(5,not(turn))) + 150*len(board.pieces(6,not(turn)))
    val = val - len(board.pieces(1,turn)) - 3*len(board.pieces(2,turn)) - 3*len(board.pieces(3,turn)) - 5*len(board.pieces(4,turn)) - 9*len(board.pieces(5,turn)) - 150*len(board.pieces(6,turn))
    
    """ Bonus for Bishop Pair """
    if (len(board.pieces(3,turn)) == 2):
        val = val - 0.2
    if (len(board.pieces(3,not(turn))) == 2):
        val = val + 0.2
    return val

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

def EPTplayout(startboard,EarlyStop):
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
    return val

def ADEPTplayout(startboard,EarlyStop):
    """ Takes a board as an inputs,and a number of moves after which will terminate the simulation and apply the Valuation Function.
    and then plays a game from the board position by playing random moves until termination or game completion. """
    board = chess.Board(startboard.fen())
    currentVal = ETPValFunc(board)
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
            if ETPValFunc(board) > currentVal:
                val = 1
            else:
                val = 0
        else:
            if ETPValFunc(board) < currentVal:
                val = 1
            else:
                val = 0
    else:        
        if (board.result() == '1/2-1/2'):
            val = 0.5
        elif (board.result() =='1-0'):
            val = 1
        else:
            """ !=draw & !=win => loss"""
            val = 0
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
        selectedMove = sorted(self.childNodes, key = lambda c: c.wins/(c.playouts + epsilon) + math.sqrt(2*math.log(self.playouts)/(c.playouts+epsilon)))[-1]
        
        return selectedMove
        
    def SelectChildValFunc(self):
        selectedMove = sorted(self.childNodes, key = NodeValFunc)[-1]
        
        return selectedMove    
    def AddChild(self, m, board):
        """
        Remove the move from vUntriedMoves and add a new child node 
        """
        newNode = Node(move = m, parent = self, board = board)
        self.vUntriedMoves.remove(m)
        self.childNodes.append(newNode)
        return newNode

def ADAPTMCTS(rootfen,itermax, MCTypeFlag,verbose = False):
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
        if MCTypeFlag == 0:
            result = playout(chess.Board(board.fen()))
        elif MCTypeFlag == 1:
            result = EPTplayout(chess.Board(board.fen()),5)
        elif MCTypeFlag == 2:
            result = ADEPTplayout(chess.Board(board.fen()),5)
        else:
            'ERROR'
        
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

def SimulateMatch2(AlgoA,AlgoB):
    """ Takes input arguments of the maximum num of iterations for White and Black"""
    Algo1, p1 = AlgoA
    Algo2, p2 = AlgoB
    
    w = 0
    d = 0
    l = 0
    board = chess.Board()
    
    if Algo1 == 'MM':
        if Algo2 == 'MCTS':
            while (not board.is_game_over(claim_draw=False)):
                if (board.turn):
                    board.push_uci(calcMinimaxMove(board,p1,board.turn,float("-inf"),float("inf"))[1].uci())
                else:
                    board.push_uci(ADAPTMCTS(board.fen(),p2,0).uci())
        elif Algo2 == 'EPT':
            while (not board.is_game_over(claim_draw=False)):
                if (board.turn):
                    board.push_uci(calcMinimaxMove(board,p1,board.turn,float("-inf"),float("inf"))[1].uci())
                else:
                    board.push_uci(ADAPTMCTS(board.fen(),p2,1).uci())
        elif Algo2 == 'ADEPT':
            while (not board.is_game_over(claim_draw=False)):
                if (board.turn):
                    board.push_uci(calcMinimaxMove(board,p1,board.turn,float("-inf"),float("inf"))[1].uci())
                else:
                    board.push_uci(ADAPTMCTS(board.fen(),p2,2).uci())               
    elif Algo1 == 'MCTS':
        if Algo2 == 'MM':
            while (not board.is_game_over(claim_draw=False)):
                if (board.turn):
                    board.push_uci(ADAPTMCTS(board.fen(),p1,0).uci())
                else:
                    board.push_uci(calcMinimaxMove(board,p2,board.turn,float("-inf"),float("inf"))[1].uci())
        elif Algo2 == 'EPT':
            while (not board.is_game_over(claim_draw=False)):
                if (board.turn):
                    board.push_uci(ADAPTMCTS(board.fen(),p1,0).uci())
                else:
                    board.push_uci(ADAPTMCTS(board.fen(),p2,1).uci())
        elif Algo2 == 'ADEPT':
            while (not board.is_game_over(claim_draw=False)):
                if (board.turn):
                    board.push_uci(ADAPTMCTS(board.fen(),p1,0).uci())
                else:
                    board.push_uci(ADAPTMCTS(board.fen(),p2,2).uci())
    elif Algo1 == 'EPT':
        if Algo2 == 'MM':
            while (not board.is_game_over(claim_draw=False)):
                if (board.turn):
                    board.push_uci(ADAPTMCTS(board.fen(),p1,1).uci())
                else:
                    board.push_uci(calcMinimaxMove(board,p2,board.turn,float("-inf"),float("inf"))[1].uci())
        elif Algo2 == 'MCTS':
            while (not board.is_game_over(claim_draw=False)):
                if (board.turn):
                    board.push_uci(ADAPTMCTS(board.fen(),p1,1).uci())
                else:
                    board.push_uci(ADAPTMCTS(board.fen(),p2,0).uci())
        elif Algo2 == 'ADEPT':
            while (not board.is_game_over(claim_draw=False)):
                if (board.turn):
                    board.push_uci(ADAPTMCTS(board.fen(),p1,1).uci())
                else:
                    board.push_uci(ADAPTMCTS(board.fen(),p2,2).uci())
    elif Algo1 == 'ADEPT':
        if Algo2 == 'MM':
            while (not board.is_game_over(claim_draw=False)):
                if (board.turn):
                    board.push_uci(ADAPTMCTS(board.fen(),p1,2).uci())
                else:
                    board.push_uci(calcMinimaxMove(board,p2,board.turn,float("-inf"),float("inf"))[1].uci())
        elif Algo2 == 'MCTS':
            while (not board.is_game_over(claim_draw=False)):
                if (board.turn):
                    board.push_uci(ADAPTMCTS(board.fen(),p1,2).uci())
                else:
                    board.push_uci(ADAPTMCTS(board.fen(),p2,0).uci())
        elif Algo2 == 'EPT':
            while (not board.is_game_over(claim_draw=False)):
                if (board.turn):
                    board.push_uci(ADAPTMCTS(board.fen(),p1,2).uci())
                else:
                    board.push_uci(ADAPTMCTS(board.fen(),p2,1).uci())
           
    if (board.result() == '1/2-1/2'):
        d = 1
    elif(board.result() == '1-0'):
        w = 1
    else:
        l = 1
    return w,d,l

def testAlgorithms(a1,p1,a2,p2,Q):
    """Plays equal number of games as white and black with the two inputted algorithms, returns the results."""
    w=0
    d=0
    l=0
    for i in range(math.ceil(Q/2)):
        nw, nd, nl = SimulateMatch2((a1,p1),(a2,p2))
        w +=nw 
        d +=nd
        l += nl
    
    for i in range(math.ceil(Q/2),Q):
        nl, nd, nw = SimulateMatch2((a2,p2),(a1,p1))
        w +=nw 
        d +=nd
        l += nl
        
    print('Win % ' + str(w/len(range(Q))))
    print('Draw % '+ str(d/len(range(Q))))
    print('Loss % '+ str(l/len(range(Q))))
    return 'Finished'
  
#testAlgorithms('ADEPT',1,'EPT',1,100)
#print("------")
#testAlgorithms('ADEPT',5,'EPT',5,100)
#print("------")
#testAlgorithms('ADEPT',10,'EPT',10,100)
#print("------")
print("depth 2 plain Minimax v.s 51 Playout Simulations")
testAlgorithms('MM',2,'ADEPT',51,200)
print("------")
#testAlgorithms('ADEPT',30,'EPT',30,200)
#print("------")
#testAlgorithms('ADEPT',60,'EPT',60,200)
print("depth 3 plain Minimax v.s 1400 Playout Simulations")
testAlgorithms('MM',3,'ADEPT',1400,200)
print("------")
#testAlgorithms('ADEPT',120,'EPT',120,100)

      
      
      
      
