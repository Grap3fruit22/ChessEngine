# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:22:04 2019

@author: 44775
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:33:03 2019

@author: 44775
"""
""" Constructs an opening book from a mixture of MCTS, random games, Minimax and human games."""
""" Automaticaly terminantes if memory usage is too high."""

import os
import math
import chess
import random
import pickle

""" Reading and constructing implementation of human games ---------- """

def NodeConverter(Node):
    """Converts a node into a linked list equivalent for saving and storage purposes."""  
    return [Node.move,Node.wins,Node.playouts,[NodeConverter(cnode) for cnode in Node.childNodes]]

def ReadData(filename):
    """Opens a PGN file containing Human games from Lichess and extracts the raw data. Clear everything 
    which isn't, the game in moves, the result, termination, Black Elo, White Elo. 
    Also add empty fields to enfore structure if missing items."""

    text_file = open(filename,"r")
    RawLines = text_file.readlines()
    
    termCount = 0
    CleanLines = []
    cleaningLines = []
    IndexingBox = []
    
    for line in RawLines:
    
        if 'Result ' in line:
            cleaningLines.append(line[9:12])
        else:    
            if 'WhiteElo' in line:
                cleaningLines.append(line[line.find('"')+1:line.rfind('"')])
            else:
                if 'BlackElo' in line:
                    cleaningLines.append(line[line.find('"')+1:line.rfind('"')])
                else:
                    if 'Termination' in line:
                        termCount += 1
                        cleaningLines.append(line[line.find('"')+1:line.rfind('"')])
                        IndexingBox.append(len(cleaningLines))
                    else:
                        if '1. ' in line:
                            cleaningLines.append(line)        

    tcount = 0
    for counter in range(len(cleaningLines)):

        if (counter in IndexingBox) and (tcount < termCount-1):
            tcount += 1
            if (IndexingBox[tcount]-IndexingBox[tcount-1] < 5):
                CleanLines.append('EmptyField')

        CleanLines.append(cleaningLines[counter])

    return CleanLines
      
def CleanDataSet(RawData,MinimumElo,EloTolerance):
    """Removes games which terminate from time or abandons, and games. rated below a threshold."""
    CleanData = []
    for num in range(int(len(RawData)/5)-1):  
            if (int(RawData[num*5+1]) < MinimumElo) or (int(RawData[num*5+2]) < MinimumElo) or (abs(int(RawData[num*5+1])-int(RawData[num*5+2])) > EloTolerance):
                """Delete because Elo below Threshold, or discrepency between players is too high."""
                continue
            else:
                if (RawData[num*5+3] == 'Abandoned') or (RawData[num*5+3] == 'Time forfeit'):
                    """Delete because of terminated from time or abandoned."""
                    continue
                else:
                    if RawData[num*5+4] == 'EmptyField':
                        """Remove instantly resigned games"""
                        continue
                    else:
                        """ Otherwise keep the game."""
                        CleanData.append(RawData[num*5])
                        CleanData.append(RawData[num*5+1])
                        CleanData.append(RawData[num*5+2])
                        CleanData.append(RawData[num*5+3])
                        CleanData.append(RawData[num*5+4])
                        if (num == int(len(RawData)/5)-1):
                            CleanData.append(RawData[num*5+5])
    return CleanData  

def ExtractAndPreprocess(filename,EloRange,EloTolerance):
    """Reads raw data from file and cleans it.""" 
    RawData = ReadData(filename)
    Data = CleanDataSet(RawData,EloRange,EloTolerance)
    
    return Data

def cleanMoves(GameString):
    """ Takes a string of moves, and removes the junk."""
    
    """Clean Stockfish evaluation punctuation"""
    #GameString.translate(str.maketrans('', '', string.punctuation))
    GameString = GameString.replace('???','').replace('?','').replace('?','').replace('!','').replace('+','').replace('!?','').replace('?!','')
    
    """Split into individual moves, and remove stockfish evaluations"""
    RawMoveset = GameString.split()
    Moveset = []
    for move in RawMoveset:
        if move[0].isdigit():
            continue
        else:
            if move[0] == '{':
                continue
            else:
                if move[0] =='}':
                    continue
                else:
                    if move[0] == '[':
                        continue
                    else:
                        if move[0] == '-':
                            continue
                        else:
                            if move[0] == '#':
                                continue
                            else:
                                Moveset.append(move)
        
    return Moveset
        
def etaFunction(InWeight):
    """ This function takes in weights of human games and scales them up."""
    OutWeight =100*InWeight
    
    return OutWeight

""" ---------- Node Related Code  -----------"""
def ResultValue(Boardresult):
    if (Boardresult == '1/2-1/2'):
        val = 0.5
    else:
        if (Boardresult =='1-0'):
            val = 1
        else:
            val = 0
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

def ETPValFunc(board):
    """ Valuation function that takes a board position and outputs the value of position.
    Is slimmed down compared to the Minimax valuation function, because Lorentz suggested
    faster less domain knowledge dependent valuation function performed better.
    This makes intuitive sense because the valuation function is triggered far more, due to
    monte carlo tree search sims."""
    phase = (len(board.pieces(5,True))*8+len(board.pieces(5,False)))*8+(len(board.pieces(3,True))*3+len(board.pieces(3,False)))*3 + (len(board.pieces(2,True))*3+len(board.pieces(2,False)))*3+(len(board.pieces(4,True))*5+len(board.pieces(4,False)))*5 + (len(board.pieces(1,True))+len(board.pieces(1,False)))
    val = (phase/68)*SimpleMidgameFunc(board) + (68-phase)/68*SimpleEndgameFunc(board)

    return val
             
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
        val = ResultValue(board.result())
        
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
        
    def Update(self,value,multiplier):
        self.playouts += 1*multiplier
        self.wins += value*multiplier
        
    def SelectChildPureMC(self):
        """ Selects the move based on the UCT formula. """     
        selectedMove = sorted(self.childNodes, key = lambda c: 1/c.playouts)[-1]
        
        return selectedMove
    
    def AddChild(self, m, board):
        """
        Remove the move from vUntriedMoves and add a new child node 
        """
        newNode = Node(move = m, parent = self, board = board)
        self.vUntriedMoves.remove(m)
        self.childNodes.append(newNode)
        return newNode
    
""" --------------------- Constructing the Game Tree from Human Games ---------------"""

def UpdateGameTree(rtNode,GameMoves,result,elo):
    """ Adds first 10 moves to the tree, or the full game, if the game length is less than 10 moves,
    Updates the visist and wins at each node, multiplied by the eta function, in this case EtaMult."""
    EtaMult = math.floor(float(elo)**3/160000)
    
    """ These variables track the current node and current board."""
    Currentboard = chess.Board()
    Cnode = rtNode
    
    for num in range(min(16,len(GameMoves))):
        """ Loop through child nodes updating values multiplied by eta constant."""
        m = GameMoves[num]
        Currentboard.push_san(m)
        
        if not(m in Cnode.vUntriedMoves):
            """Select the child"""
            for node in Cnode.childNodes:
                if (node.move == Currentboard.peek()):
                    Cnode = node
                    break
        else:
            """Add the child as it doesn't exist, and it, and select it."""
            
            Cnode = Cnode.AddChild(Currentboard.peek(),Currentboard)
        """ Update the node """
        if (num % 2 == 0):
            """ If white update the result"""
            Cnode.Update(result,EtaMult)
        else:
            """Otherwise update the inverted result."""
            Cnode.Update(1-result,EtaMult)
        
    return rtNode

def ConstructGameTree(MinElo,EloTol):
    """ Loops over all PGNs in directory, extracts then preprocesses data. Finally constructs a tree from the games,
    and updates the visits and wins at each node, ready for merging with MCTS."""
    """Only takes games where both players are above 1450 elo"""
    
    GameTree = Node(board = chess.Board())
    counter = 0
    totgames = 0
    
    directory = os.fsencode("C:/Users/44775/Documents/lichess_Games_2017_01/")
    for file in os.listdir(directory):
        #display(str(counter) + ' / 939')
        filename = "C:/Users/44775/Documents/lichess_Games_2017_01/" + os.fsdecode(file)
        #if filename.endswith(".???"):
        print('Analysing games... ')
        RawData = ReadData(filename)
        Data = CleanDataSet(RawData,MinElo,EloTol)
        totgames += len(Data)/5
        for num in range(int(len(Data)/5)):
            Moves = cleanMoves(Data[5*num+4])
            result = ResultValue(Data[5*num])
            GameTree = UpdateGameTree(GameTree,Moves,result,(float(Data[5*num+1])+float(Data[5*num+2]))/2)
        counter += 1
        print('Complete. Total games analyzed = ' + str(totgames))
    return GameTree
    
def ConstructBook(filename,itermax, DepthCutout,  verbose = False):
    count = -5;
    rootfen = chess.Board().fen()
    
    """ Initializes the tree with human games, and propogates them up the tree."""
    MinElo = 0
    EloTolerance = 800
    
    """rootNode = ConstructGameTree(MinElo,EloTolerance)"""
    rootNode = Node(board = chess.Board())
    
    """ Performs itermax number of ETP simulations"""
    for i in range(itermax):
        node = rootNode
        board = chess.Board(rootfen).copy()
        
        """SELECTION, important to prevent selection of nodes below depth 10."""
        while (node.vUntriedMoves == []) and (node.childNodes != []):
            node = node.SelectChildPureMC()
            board.push_uci(node.move.uci())
        
        """EXPAND"""
        if node.vUntriedMoves != []:
            m = random.choice(node.vUntriedMoves)
            board.push_uci(m.uci())
            node = node.AddChild(m,chess.Board(board.fen()))
            
        """ SIMULATION """
        result = ETPplayout(chess.Board(board.fen()), DepthCutout)
        
        """BACKPROPOGATION"""
        while node != None:
            if node.col:
                node.Update(result,1)
                node = node.parentNode
            else:
                node.Update(1-result,1)
                node = node.parentNode
                
        if (i % (itermax/20)) == 0:
            count +=5
            print(str(count) + ' % Complete.')
        
    assert os.path.exists(filename), 'File does not exist.'
    """Save linked list to Memory using Pickle:"""
    file_OB1 = open(filename, 'wb')
    pickle.dump(NodeConverter(rootNode),file_OB1)
        
    return 'Opening Book Created.'
        
#filename = "C:/Users/44775/Documents/lichess_Games_2017_01/lichess_db_standard_rated_2017-01.10.pgn"
#CompleteRawData = ReadData(filename)
#Data = CleanDataSet(CompleteRawData,1300,225)

#cMoves = cleanMoves(Data[4])
#rtNode = UpdateGameTree(rootNode,cMoves,1)
    
"""Construct directry name, as unknown."""
FileDir = os.path.dirname(os.path.abspath(__file__))
print(FileDir)

filename = FileDir + "/LinkedBk1.txt"
if os.path.exists(filename):
    os.remove(filename)
    print('Old opening book has been deleted')

file_OB = open(filename,'x')

ConstructBook(filename,35000,10)



