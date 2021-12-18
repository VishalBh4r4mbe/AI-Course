import numpy as np
from Board import Board 
import random
from timer import Timer
class SimulatedAnnealing:
    def __init__(self,n,iterations,temparature,coolingFactor, board=None):
        self.board = Board(n, board)
        self.iterations = iterations
        self.temparature = temparature
        self.coolingFactor = coolingFactor
        self.n = n
    
    def getSolution(self):
        curScore = self._getHueristicScore(self.board.getState())
        for _ in range(self.iterations):
            newState = self.nextState()
            curScore = self._getHueristicScore(newState)
            # print(curScore)
            if curScore== self.n*(self.n - 1):
                self.board.setState(newState)
                self.board.printBoard()
                return True
            self.temparature = max(self.temparature*self.coolingFactor,0.001)
            self.board.setState(newState)  


        return False
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
    def nextState(self):
        startCost = self._getHueristicScore(self.board.getState())
        while(True):
            nextRow = random.randint(0,self.n-1)
            nextCol = random.randint(0,self.n-1)
            curState = self.board.getState()
            tempStore = curState[nextRow]
            curState[nextRow] = nextCol
            if self._getHueristicScore(curState) >startCost:
                return curState
            else:
                dE = self._getHueristicScore(curState) - startCost
                # print("dE = ",dE,", temp = ",self.temparature)
                p = np.exp(dE/self.temparature)
                # print("p",p)
                if(random.random()<p):
                    return curState
                else:
                    curState[nextRow] = tempStore

runs = 200
success = 0
ti = Timer()
# x = SimulatedAnnealing(8,10000,100,0.999) #99%
# # # print(x._getHueristicScore([7, 1, 6, 2, 0, 6, 3, 5]))
# print(x.getSolution())
x = SimulatedAnnealing(8,10000,100,0.999)
isPossible = x.getSolution()

# for _ in range(runs):
#     x = SimulatedAnnealing(20,10000,100,0.999)
#     ti.start()
#     if x.getSolution():
#         success+=1
#     if _%10==0 and _!=0:
#         print("Runs = ", _, ",Success rate: ",success/(_+1),", Time: ",ti.stop())
    
# print("Success rate: ",success/runs,", Time: ",ti.avg_time())
    
