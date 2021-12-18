from CNF_Creator import *
from functools import cmp_to_key
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
'''Basically for benchmarking'''
class Timer:
    def __init__(self):
        self.times = [] #for maintaing the times on the same Timer Object
        self.start()
    def start(self):
        self.initial = time.time()
    def timePeak(self):
        return time.time() - self.initial
    def stop(self):
        self.times.append(time.time() - self.initial)
        return self.times[-1]

    def avg_time(self):
        return np.mean(self.times)
    def sum(self):
        return np.sum(self.times)
    def reset(self):
        self.times = []
class CNFEvaluator:

    def getHueristicScore(clause,assignment):
        score = 0
        for part in clause:
            if CNFEvaluator.satisfied(part,assignment):
                score += 1
        return score
    
    def satisfied(clause,assignment):
        for part in clause:
            if(part == assignment[abs(part)-1]):
                """ -1 is there because the passed assignment has 1 based indexing"""
                return True
        return False
class CNFSolverGenetic:
    def __init__(self,sentence,n , size , mutationRate,generations):
        self.sentence = sentence
        self.n = n
        self.size = size
        self.mutationRate = mutationRate
        self.generations = generations
        assert(self.size%2==0)
    def getComparatorandScorer(sentence):
        def compare(x, y):
            if CNFEvaluator.getHueristicScore(sentence,x)>CNFEvaluator.getHueristicScore(sentence,y):
                return 1
            elif CNFEvaluator.getHueristicScore(sentence,x)<CNFEvaluator.getHueristicScore(sentence,y):
                return -1
            else:
                return 0
        def scoreCalculator(assignment):
            return CNFEvaluator.getHueristicScore(sentence,assignment)
        return compare,scoreCalculator
    def getSolution(self):
        self.tim = Timer()
        self.states=[]
        comparator,scoreCalculator = CNFSolverGenetic.getComparatorandScorer(self.sentence)
        self.tim.start()
        for i in range(self.size):
            curState=[]
            for j in range(self.n):
                randInt = random.randint(0,1)
                if(randInt==0):
                    curState.append(-(j+1))
                else:
                    curState.append(j+1)
            self.states.append(curState)
        for i in range(self.generations):
            setOfStates = set(tuple(state)for state in self.states)
            self.states = list(list(state) for state in setOfStates)
            while len(self.states)<self.size:
                #print("random pop injected in generation", i)
                self.states.append([random.choice([-(i+1),(i+1)]) for i in range(self.n)])
            self.states = sorted(self.states,key=cmp_to_key(comparator),reverse=True)
            #print("Generation",i,":",scoreCalculator(self.states[0]))
            if(self.tim.timePeak()>44):
                #print(self.states[0])
                return (False,self.states[0])
            """Crossover"""
            for j in range(int(self.size/2)):
                """Get random crossover point and get crossover states"""
                self.states[2*j],self.states[2*j+1] = CNFSolverGenetic.getCrossoverStates(self.states[2*j],self.states[2*j+1])
            """Mutation"""
            for i in range(self.size):
                if scoreCalculator(self.states[i]) == len(self.sentence):
                    return (True,self.states[i])
                if random.random()<self.mutationRate:
                    state = CNFSolverGenetic._mutate(self.tim,self.states[i],self.sentence)
                    if scoreCalculator(self.states[i]) == len(self.sentence):
                        return (True,self.states[i])
        self.states = sorted(self.states,key=cmp_to_key(comparator),reverse=True)
        return (False,self.states[0])
    def _mutate(tim,state,sentence,type = 'mbgreedy'):
        if type == 'greedy':
            originalScore = CNFEvaluator.getHueristicScore(sentence,state)
            count = 0
            while True:
                count += 1
                randInt = random.randint(0,len(state)-1)
                temp = state[randInt]
                state[randInt] = -state[randInt]
                if CNFEvaluator.getHueristicScore(sentence,state) >originalScore:
                    return state
                if(count>=len(state)):
                    return [random.choice([-(i+1),(i+1)]) for i in range(len(state))]
                state[randInt] = temp
            return state
        elif type == 'mbgreedy':
            originalScore = CNFEvaluator.getHueristicScore(sentence,state)
            count = 0
            for i in range(len(state)):
                cur =state[i]
                state[i] = -state[i]
                if CNFEvaluator.getHueristicScore(sentence,state) >originalScore:
                    originalScore = CNFEvaluator.getHueristicScore(sentence,state)
                    continue
                else :
                    state[i] = cur
            if(count==0):
                return [random.choice([-(i+1),(i+1)]) for i in range(len(state))]
            return state
    
    def getCrossoverStates(state1,state2):
        crossOverPoint = random.randint(0,len(state1)-1)
        newState1 = state1[crossOverPoint:]
        state1[crossOverPoint:]=state2[crossOverPoint:]
        state2[crossOverPoint:]=newState1
        return state1,state2


def main():
    cnfC = CNF_Creator(n=50) # n is number of symbols in the 3-CNF sentence
    #sentence = cnfC.CreateRandomSentence(m=5000) # m is number of clauses in the 3-CNF sentence
    # print('Random sentence : ',sentence)
    #sentence = cnfC.ReadCNFfromCSVfile()
    #print('length of sentence : ',len(sentence))
    # print('\nSentence from CSV file : ',sentence)
    #t = Timer()
    #t.start()
    mVal = 100
    m_s=[]
    history=[]
    numberOfTimes=50
    times=[]
    while (mVal<=300):
        m_s.append(mVal)
        sumFitness=0
        t = Timer()
        for i in tqdm(range(numberOfTimes)) :
            t.start()
            sentence = cnfC.CreateRandomSentence(m=mVal)
            solver = CNFSolverGenetic(sentence,n=50,size=50,mutationRate=1,generations=500)
            x = solver.getSolution()
            score = CNFEvaluator.getHueristicScore(sentence,x[1])
            sumFitness+=score
            t.stop()
        averageFitness = sumFitness/(numberOfTimes*mVal)
        history.append(averageFitness)
        times.append(t.avg_time()) 
        print(f"for m = {mVal}, average fitness = {averageFitness} , avg_time = {t.avg_time()}")
        mVal+=20
    figure , axis = plt.subplots(1,2)
    print(m_s)
    print(history)
    axis[0].plot(m_s,history)
    axis[0].set_title("Fitness vs m")
    axis[0].set_xlabel("m")
    axis[0].set_ylabel("Fitness")
    axis[1].plot(m_s,times)
    axis[1].set_title("Time vs m")
    axis[1].set_xlabel("m")
    axis[1].set_ylabel("Time")
    plt.show()
    #solver = CNFSolverGenetic(sentence,n=50,size=50,mutationRate=1,generations=500)
    #x = solver.getSolution()
    # x= [-1,-2,-3,-4,5,-6]
    # y= [1,2,3,4,5,6]
    # CNFSolverGenetic.getCrossoverStates(x,y)
    # print(x,y)
    #print('Solution : ',x[1])
    #print('Hueristic Score : ',CNFEvaluator.getHueristicScore(sentence,x[1]))
#    print('\n\n')
#    print('Roll Nos : 2020H1030999G')
#    print('Number of clauses in CSV file : ',len(sentence))
#    print('Best model : ',[1, -2, 3, -4, -5, -6, 7, 8, 9, 10, 11, 12, -13, -14, -15, -16, -17, 18, 19, -20, 21, -22, 23, -24, 25, 26, -27, -28, 29, -30, -31, 32, 33, 34, -35, 36, -37, 38, -39, -40, 41, 42, 43, -44, -45, -46, -47, -48, -49, -50])

#    print('Fitness value of best model : 99%')
    #print('Time Taken :', t.stop(),' seconds')
#    print('Time taken : 5.23 seconds')
#    print('\n\n')
    
if __name__=='__main__':
    main()