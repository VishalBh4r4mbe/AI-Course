from Board import  Board
from functools import cmp_to_key
class LocalBeamSearch:
    def __init__(self,n, iterations,numberOfStates):
        self.n=n
        self.iterations = iterations
        self.numberOfStates = numberOfStates
        self.states = []
        for i in range(numberOfStates):
            b = Board(n)
            self.states.append(b.getState())

    
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

    def comparator(state1,state2):
        if LocalBeamSearch._getHueristicScore(state1)>LocalBeamSearch._getHueristicScore(state2):
            return 1
        elif LocalBeamSearch._getHueristicScore(state1)==LocalBeamSearch._getHueristicScore(state2):
            return 0
        else:
            return -1

    def getSolution(self):
        for i in range(self.iterations):
            nextStates = []
            for state in self.states:
                curFitnessScore = LocalBeamSearch._getHueristicScore(state)
                if curFitnessScore == self.n*(self.n-1):
                    return (True,state)
                else:
                    for i in range(self.n):
                        nextStates.append(self.getNextState(state.copy(),i,curFitnessScore))
            self.states = sorted(nextStates,key=cmp_to_key(LocalBeamSearch.comparator),reverse=True)
            self.states = self.states[:self.numberOfStates]
        return  (False,None)
    def getNextState(self,state, row, previousFitness):
        for i in range(self.n):
            if i == state[row]:
                continue
            else:
                lastCol = state[row]
                state[row] = i
                nextFitness = LocalBeamSearch._getHueristicScore(state)
                if nextFitness > previousFitness:
                    return state
                else:
                    state[row] = lastCol               
        x= Board(len(state))
        return x.getState()


x = LocalBeamSearch(20,1000,10)
isPossible,solution = x.getSolution()
if isPossible:
    print("Solution Found")
    x = Board(len(solution),solution)
    x.printBoard()