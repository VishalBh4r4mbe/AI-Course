# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 18:04:23 2021

@author: visha
"""
import numpy as np
from Board import Board 



class HillClimbing:
    def __init__(self,n, board = None):
        """

        Parameters
        ----------
        n : int
            length of side of the board.
        board : list[int], optional
            initial configuration of board, if left None, it is randomly initialised. The default is None.

        Returns
        -------
        None.

        """
        self.board = Board(n,board)
        self.n = n

    def _getHueristicScore(self,state):
        """

        Parameters
        ----------
        state : list[int]
            board.

        Returns
        -------
        int
            hueristic value chosen.

        """
        cost = 0
        n = len(state)
        for i in range(len(state)):
            for j in range(len(state)):
                if i==j:
                    continue
                else:
                    if state[i]==state[j] or state[i]-state[j] == i-j or i+ state[i]==state[j]+j:
                        cost+=1;         
        return self.n*(self.n - 1) - cost
    
    def getSolution(self):
        """

        Returns
        -------
        bool
            if the current search resulted in finding a solution.

        """
        curScore = self._getHueristicScore(self.board.getState())
        while(True):
            # print("Board------",)
            curScore = self._getHueristicScore(self.board.getState())
            # print(self.board.board , "Score : ",curScore)
            nextState = self._getNextState()
            if(curScore==self._getHueristicScore(nextState)):
                break
            else:
                self.board.setState(nextState)
        if curScore!= self.n*(self.n -1):
            return False
        else:
            # self.board.printBoard()
            return True
    def _getNextState(self):
        """
        Returns
        -------
        bestValueState : list[int]
            board with the next best value.

        """
        bestValue = self._getHueristicScore(self.board.getState())
        bestValueState = self.board.getState()
        initState = self.board.getState()
        for i in range(self.n):
            curState = initState.copy()
            for j in range(self.n):
                if j==initState[i]:
                    continue
                else:
                    curState[i] =j
                nowScore = self._getHueristicScore(curState)
                if(nowScore>bestValue):
                    # print("updating from", bestValue ,"to ",nowScore)
                    bestValue=nowScore
                    bestValueState = curState.copy()
            
        return bestValueState


x = HillClimbing(8)

runs = 10000
successfulRuns =0
for i in range(runs):
    x = HillClimbing(8)
    poss = x.getSolution()
    if(poss):
        successfulRuns+=1
    if(i%1000==0 and i!=0):
        print("Runs : ",i,"Success rate = ", successfulRuns/i * 100)

print("successful runs = ",successfulRuns, "out of ", runs)