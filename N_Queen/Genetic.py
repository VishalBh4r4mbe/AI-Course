from Board import Board
import random 
from functools import cmp_to_key
class GeneticAlgorithm():
    def __init__(self,n, size , mutationRate,generations):
        assert(size%2==0)
        self.n = n
        self.size = size
        self.mutationRate = mutationRate
        self.generations = generations
        self.states = []
        for i in range(size):
            b = Board(n)
            self.states.append(b.getState())
    def comparator(state1,state2):
        if GeneticAlgorithm._getHueristicScore(state1)>GeneticAlgorithm._getHueristicScore(state2):
            return 1
        elif GeneticAlgorithm._getHueristicScore(state1)==GeneticAlgorithm._getHueristicScore(state2):
            return 0
        else:
            return -1
    def getSolution(self):
        for i in range(self.generations):
            self.states = sorted(self.states,key=cmp_to_key(GeneticAlgorithm.comparator),reverse=True)
            """Crossover"""
            for i in range(int(self.size/2)):
                """get random crossover point and seggs"""
                self.states[i],self.states[2*i+1] = GeneticAlgorithm.getCrossoverStates(self.states[2*i],self.states[2*i+1])
            """Mutation"""
            for i in range(self.size):
                if GeneticAlgorithm._getHueristicScore(self.states[i]) == self.n*(self.n-1):
                    return (True,self.states[i])
                if random.random()<self.mutationRate:
                    state = GeneticAlgorithm._mutate(self.states[i])
                if GeneticAlgorithm._getHueristicScore(self.states[i]) == self.n*(self.n-1):
                    return (True,state)
        return (False,None)
    def _mutate(state):
        state[random.randint(0,len(state)-1)] = random.randint(0,len(state)-1)
        return state
    
    def getCrossoverStates(state1,state2):
        crossOverPoint = random.randint(1,len(state1)-1)
        newState1 = state1[crossOverPoint:]
        state1[crossOverPoint:]=state2[crossOverPoint:]
        state2[crossOverPoint:]=newState1
        return state1,state2
      
    def _getHueristicScore(state):
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
        return n*(n - 1) - cost


x = GeneticAlgorithm(8,210,0.3,1000)

poss,b = x.getSolution()
if poss:
    board = Board(8,b)
    board.printBoard()
else :
    print("No Solution")    