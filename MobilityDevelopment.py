# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:22:51 2019

@author: 44775
"""

""" New development for working out piece mobility."""
import chess

board = chess.Board()



"""Encodes the possible movements for each piece."""
PosEffect = {1:['UL','UR'],
             2:['UUL','UUR','RRU','RRD','DDL','DDR','LLU','LLD'],
             3:['UL','UR','DL','DR'],
             4:['U','L','D','R'],
             5:['UL','U','UR','R','DR','D','DL','L']}

def PieceMobilityCounter(board,pieceFlag,col):
    
    validSqr = 0
    
    if pieceFlag == 2:
        """Do Knight Moves."""
        for piece in board.pieces(pieceFlag,col):
                for direction in PosEffect[pieceFlag]:

                    newsq = piece                        
                    if direction == 'UUL':
                        if piece > 48 or piece % 8 == 0:
                            continue
                        validSqr += 1
                    elif direction == 'UUR':
                        if piece > 48 or piece % 8 == 7:
                            continue
                        validSqr += 1
                    elif direction == 'RRU':
                        if piece % 8 > 5 or piece > 56:
                            continue
                        validSqr += 1
                    elif direction == 'RRD':
                        if piece % 8 > 5 or piece < 7:
                            continue
                        validSqr += 1
                    elif direction == 'DDL':
                        if newsq < 15 or newsq % 8 > 6:
                            continue
                        validSqr += 1
                    elif direction == 'DDR':
                        if newsq < 15 or newsq % 8 > 6:
                            continue
                        validSqr += 1
                    elif direction == 'LLU':
                        if newsq > 56 or newsq % 8 < 2:
                            continue
                        validSqr += 1
                    elif direction == 'LLD':
                        if newsq < 7 or newsq % 8 < 2:
                            continue
                        validSqr += 1
                    else:
                        print('error')
        
    else:

        """These pieces more in *Rays* which extend across the board thus we apply poss directions repeatedly."""
        for piece in board.pieces(pieceFlag,col):
            
            for direction in PosEffect[pieceFlag]:

                newsq = piece
                if direction == 'U':
                    while newsq < 56:
                        newsq += 8    
                        validSqr += 1
                    continue
                    
                elif direction == 'L':
                    while newsq % 8 != 0:
                        newsq += -1
                        validSqr += 1
                    continue
                
                elif direction == 'R':
                    while newsq % 8 != 7:
                        newsq += 1
                        validSqr += 1
                    continue
                
                elif direction == 'D':
                    while newsq > 7:
                        newsq += -8
                        validSqr += 1
                    continue
                    
                elif direction == 'UR':
                    while newsq < 56 and newsq % 8 != 7:
                        newsq += 9
                        validSqr += 1
                    continue
                        
                elif direction == 'DR':
                    while newsq > 7 and newsq % 8 != 7:
                        newsq += -7
                        validSqr += 1
                    continue
                        
                elif direction == 'DL':
                    while newsq > 7 and newsq % 8 != 0:
                        newsq += -9
                        validSqr += 1
                    continue
                        
                elif direction == 'UL':
                    while newsq < 56 and newsq % 8 != 0:
                        newsq += 7
                        validSqr += 1
                    continue
                    
                else:
                    print('error: Unknown directional input.')
                
    return validSqr
        
        
X = PieceMobilityCounter(board,2,True)
Y = PieceMobilityCounter(board,3,True)
Z = PieceMobilityCounter(board,4,True)
W = PieceMobilityCounter(board,5,True)


print('Knight')
print(X)
print('Bishop')
print(Y)
print('Rook')
print(Z)
print('Queen')
print(W)






        