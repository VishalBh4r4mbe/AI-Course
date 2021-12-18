# Imports
import numpy as np
import random as rd
from copy import deepcopy
from collections import defaultdict
#Vars
INF = 100000000
RED = 2
BLACK = 1
DRAW = -1
import dill as pickle
class Board:
    def __init__(self,rows=6,columns=5):
        self.grid = np.zeros((rows, columns),dtype=np.int64)
        self.cols = [rows-1 for i in range(columns)]
        self.moves = 0
        self.lastMove = [-1, -1]
        self.rows = rows
        self.columns = columns
        self.hashValue =    np.uint64(0)
        self.MAX_MOVES = rows*columns
    
    def displayGrid(self):
        for row in self.grid:
            for element in row:
                print(element,end=' ')
            print()
    
    def makeMove(self, player, columnNumber):
        self.grid[self.cols[columnNumber]][columnNumber] = player
        self.lastMove = [self.cols[columnNumber], columnNumber]
        self.hashValue += player*np.uint64((np.power(3,columnNumber*self.rows   +  self.cols[columnNumber],dtype=np.uint64)))
        self.cols[columnNumber] =self.cols[columnNumber]- 1
        self.moves  =self.moves + 1
    
    def inBoundary(self, x, y):
        if x<0 or x>=self.rows:
            return False
        if y<0 or y>=self.columns:
            return False
        return True

    def checkVertical(self, grid, lastMove):
        x = lastMove[0]
        y = lastMove[1]
        color = grid[x][y] 
        count = 0
        while True:
            if not self.inBoundary(x, y) or color != grid[x][y]:
                return False
            x += 1
            count += 1
            if count == 4:
                return True

    def checkForWinningMove(self, color):
        bcopy = self.grid.copy()
        for i in range(7):
            if self.cols[i] >= 0:
                bcopy[self.cols[i]][i] = color
                if self.checkWinVirtual(bcopy, self.cols[i], i):
                    return i
                bcopy[self.cols[i]][i] = 0
        return -1

    def checkHorizontal(self, grid, lastMove):
        x = lastMove[0]
        y = lastMove[1]
        color = grid[x][y]
        count = 0
        while True:
            if not self.inBoundary(x, y) or color != grid[x][y]:
                break
            y += 1
            count += 1
            if count == 4:
                return True

        x = lastMove[0]
        y = lastMove[1] - 1

        while True:
            if not self.inBoundary(x, y) or color != grid[x][y]:
                break
            y -= 1
            count += 1
            if count == 4:
                return True

        if count >= 4:
            return True
        return False

    def checkDiagonal(self, grid, lastMove):
        x = lastMove[0]
        y = lastMove[1]
        color = grid[x][y]
        count = 0
        while True:
            if not self.inBoundary(x, y) or color != grid[x][y]:
                break
            x -= 1
            y += 1
            count += 1
            if count == 4:
                return True
        x = lastMove[0] + 1
        y = lastMove[1] - 1
        while True:
            if not self.inBoundary(x, y) or color != grid[x][y]:
                break
            x += 1
            y -= 1
            count += 1
            if count == 4:
                return True
        x = lastMove[0]
        y = lastMove[1]
        count = 0
        while True:
            if not self.inBoundary(x, y) or color != grid[x][y]:
                break
            x -= 1
            y -= 1
            count += 1
            if count == 4:
                return True
        x = lastMove[0] + 1
        y = lastMove[1] + 1
        while True:
            if not self.inBoundary(x, y) or color != grid[x][y]:
                break
            x += 1
            y += 1
            count += 1
            if count == 4:
                return True
        return False
    
    def isMovePossible(self, col):
        if self.cols[col] == -1:
            return False
        return True

    def checkWin(self):
        if self.checkVertical(self.grid, self.lastMove):
            return True
        if self.checkDiagonal(self.grid, self.lastMove):
            return True
        if self.checkHorizontal(self.grid, self.lastMove):
            return True
        return False

    def checkWinVirtual(self, grid, x, y):
        if self.checkVertical(grid, [x, y]):
            return True
        if self.checkDiagonal(grid, [x, y]):
            return True
        if self.checkHorizontal(grid, [x, y]):
            return True
        else:
            return False

    def resetGrid(self):
        self.grid = np.zeros((self.rows, self.columns),dtype=np.int64)
        self.cols = [self.rows-1 for i in range(self.columns)]
        self.moves = 0
        self.lastMove = [-1, -1]
    
    def getAfterStates(self,color):
        afterStates = []
        for i in range(self.columns):
            if self.cols[i] >=0:
                y = deepcopy(self)
                y.makeMove(color, i)
                afterStates.append(y)
        return afterStates
    
    def getAfterState(self,color,move):
        if self.cols[move]!=-1:
            y = deepcopy(self)
            y.makeMove(color, move)
            return y
        else:
            return None
    
    def getPossibleActions(self):
        poss = []
        for i in range(self.columns):
            if(self.cols[i]!=-1):
                poss.append(i)
        return poss

class Node:
    def __init__(self, total_score, ni, parent, state, cols, moveCnt, level,c):
        self.totalScore = total_score
        self.c = c
        self.n = ni
        self.parent = parent
        self.children = []
        self.n_actions = len(state[0])
        self.level = level
        self.state = state
        self.cols = cols
        self.moveCnt = moveCnt
        self.isTerminal = False
        self.winColor = 0

    def populateNode(self, player):
        if self.isTerminal:
            return None

        grid = Board(len(self.state), len(self.state[0]))

        for i in range(self.n_actions):
            cols = self.cols[:]
            if cols[i] == -1:  
                self.children.append(None)
                continue

            next_state = self.state.copy() 
            next_state[cols[i]][i] = player 
            node = Node(0, 0, self, next_state, cols, self.moveCnt+1, self.level + 1,self.c)
            if grid.checkWinVirtual(next_state, cols[i], i):
                node.isTerminal = True
                node.winColor = player #win for RED/BLACK
            elif node.moveCnt == grid.MAX_MOVES:
                node.isTerminal = True
                node.winColor = DRAW #draw
            node.cols[i] -= 1

            self.children.append(node)

    def calculateUCB(self, N,c=2):
        if self.n == 0:
            return INF
        ucb = (self.totalScore/self.n) + ((c)*np.log(N)/self.n)**0.5
        return ucb

    def getMaxUcbNode(self, N):
        ucbs = []

        if self.isTerminal:
            return None

        for node in self.children:
            if node:
                ucbs.append(node.calculateUCB(N))
            else:
                ucbs.append(None)

        max_ind,max_val = 0 , -1*INF

        for i in range(len(self.children)):
            if ucbs[i] != None and ucbs[i] > max_val:
                max_ind = i
                max_val = ucbs[i]

        max_node = self.children[max_ind]
        return max_node, max_ind, ucbs

    def checkLeaf(self):
        if len(self.children) == 0:
            return True
        return False

    def backpropagate(self, reward):
        self.n += 1
        self.totalScore += reward
        curNode = self.parent

        while curNode:
            curNode.n += 1
            curNode.totalScore += reward
            curNode = curNode.parent
class Agent:
    def __init__(self, color):
        self.color = color

    def getReward(self, winColor):
        if winColor == DRAW:
            return 0.5

        if self.color == winColor:
            return 1 
        return 0

    
    def makeRandomVirtualMove(self, state, cols, color):
        ok = True
        action = -1
        while ok:
            l = len(cols) 
            grid = Board(len(state),len(state[0]))
            for i in range(l):
                if cols[i] != -1:
                    state[cols[i]][i] = color
                if grid.checkWinVirtual(state, cols[i], i):
                    x = cols[i]
                    y = i
                    cols[i] -= 1
                    return state, cols, x, y
                else:
                    state[cols[i]][i] = 0 

            color = self.switchColor(color)
            for i in range(l):
                if cols[i] != -1:
                    state[cols[i]][i] = color
                if grid.checkWinVirtual(state, cols[i], i):
                    x = cols[i]
                    y = i
                    cols[i] -= 1
                    color = self.switchColor(color)
                    state[x][y] = color
                    return state, cols, x, y
                else:
                    state[cols[i]][i] = 0 
            color = self.switchColor(color)

            action = rd.randrange(l)
            if cols[action] >= 0 :
                ok = False

        state[cols[action]][action] = color
        x = cols[action]
        y = action
        cols[action] -= 1

        return state, cols, x, y

    def switchColor(self, color):
        if color == RED:
            return BLACK
        return RED


    def rollout(self, vgrid, vcols, moveCnt, colorToMove):
        grid = Board(len(vgrid),len(vgrid[0]))

        while True:
            vgrid, vcols, x, y = self.makeRandomVirtualMove(vgrid, vcols, colorToMove)
            
            moveCnt += 1
            if moveCnt == len(vgrid)*len(vgrid[0]) and not grid.checkWinVirtual(vgrid, x, y):
                return 0.5 #draw reward

            if grid.checkWinVirtual(vgrid, x, y):
                return self.getReward(colorToMove) #return win 

            colorToMove = self.switchColor(colorToMove)

    def getRewardTerminal(self, winColor):
        if winColor == DRAW:
            return 0.5

        if self.color == winColor:
            return 1 #for win
        return 0 #for loss


    def getBestMove(self, actions, n_iterations, root, grid):
        next_node = None
        action = 0
        count = 0 
        node = root
        prev_node = root
        color = BLACK

        for action in actions:
            prev_node = node

            if len(node.children) > 0:
                node = node.children[action]
            else:
                node = None
            color = self.switchColor(color)

            if not node: #check for when playing against human
                prev_node.populateNode(color)
                node = prev_node.children[action]

        if node.checkLeaf():
            node.populateNode(self.color)

        curr = node
        change = False

        while count < n_iterations:
            if not change: #to reset curr to the initial node
                curr = node
            if curr.checkLeaf():
                # print("in leaf node")
                if curr.n == 0:
                    #start rollout
                    if curr.isTerminal:
                        # print("is terminal in leaf")
                        reward = self.getRewardTerminal(curr.winColor)
                        # print("Backpropagate reward")
                        curr.backpropagate(reward)
                        
                        count += 1
                        change = False
                        continue
                    else:
                        # print("rollout in first visit")
                        vgrid = curr.state.copy()
                        vcols = curr.cols.copy()
                        colorToMove = BLACK if curr.moveCnt%2 == 1 else RED
                        
                        reward = self.rollout(vgrid, vcols, curr.moveCnt, colorToMove)
                        # print("Backpropagate reward")
                        curr.backpropagate(reward)
                        
                        count += 1
                        change = False
                        continue
                else:
                    #get node
                    colorToMove = BLACK if curr.moveCnt%2 == 1 else RED
                    # print("Expansion in visited node")

                    if curr.isTerminal:
                        # print("is terminal node ")
                        reward = self.getRewardTerminal(curr.winColor)
                        # print("Backpropagate reward")
                        curr.backpropagate(reward)
                        
                        count += 1
                        change = False
                        continue

                    curr.populateNode(colorToMove)


                    curr, _, _ = curr.getMaxUcbNode(root.n)

                    if curr.isTerminal:
                        # print("is terminal node after expansion")
                        reward = self.getRewardTerminal(curr.winColor)
                        # print("Backpropagate reward")
                        curr.backpropagate(reward)
                        
                        count += 1
                        change = False
                        continue

                    vgrid = curr.state.copy()
                    vcols = curr.cols.copy()

                    colorToMove = BLACK if curr.moveCnt%2 == 1 else RED

                    # print("Rollout in through expanded node")
                    reward = self.rollout(vgrid, vcols, curr.moveCnt, colorToMove)
                    # print("Backpropagate reward")
                    curr.backpropagate(reward)
                    
                    count += 1
                    change = False
                    continue

            else:
                change = True
                curr, _ , _= curr.getMaxUcbNode(root.n)
        _, action, _ = node.getMaxUcbNode(root.n)
        return root, action
class c4WithMCTS:
    def __init__(self, yroot, rroot, grid):
        self.yroot = yroot
        self.rroot = rroot
        self.grid = grid
        self.yroot.populateNode(RED)
        self.rroot.populateNode(RED)
        self.rAgent = Agent(RED)
        self.yAgent = Agent(BLACK)
    def play(self, n_games, n_iterations1, n_iterations2):
        yellow_wins = 0
        red_wins = 0
        for i in range(n_games):
            win = False
            # print("-----  GAME %s  -----\n"%(str(i+1)))
            actions = []
            for j in range(self.grid.MAX_MOVES):

                if j%2 == 0:  #red move
                    self.rroot, action = self.rAgent.getBestMove(actions, n_iterations1, self.rroot, self.grid)
                    actions.append(action)
                    self.grid.makeMove(RED, action)
                    # print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                    # self.grid.displayGrid()
                    if self.grid.checkWin():
                        # print("RED WINS\n")
                        red_wins += 1
                        win = True
                        break
                else: #BLACK move
                    self.yroot, action = self.yAgent.getBestMove(actions, n_iterations2, self.yroot, self.grid)
                    actions.append(action)
                    self.grid.makeMove(BLACK, action)
                    # print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                    # self.grid.displayGrid()
                    if self.grid.checkWin():
                        # print("BLACK WINS\n")
                        win = True
                        yellow_wins += 1
                        break   
            if not win:
                print("DRAW\n")
            # print("-----  GAME %s ENDS  -----\n"%(str(i+1)))        
            self.grid.resetGrid()   
        return [red_wins,yellow_wins]
class c4WithQLearning:
    def __init__(self,TDAgent):
        self.rAgent = Agent(RED)
        self.yAgent = TDAgent
    def play(self, n_games, n_iterations1):
        yellow_wins = 0
        red_wins = 0
        for i in range(n_games):
            win = False
            actions = []
            self.yAgent.reset()
            grid = Board(4,5)
            self.rroot = Node(0, 0, None, grid.grid, grid.cols, grid.moves, 0,2)
            self.rroot.populateNode(RED)    
            
            if i %1000==0:
                print(len(self.yAgent.Q.items()))
                print(red_wins,yellow_wins)
                yellow_wins,red_wins =0,0
            for j in range(grid.MAX_MOVES):
            
                if j%2 == 0:
                    self.rroot, move = self.rAgent.getBestMove(actions, n_iterations1, self.rroot, grid)
                    actions.append(move)
                    grid.makeMove(RED, move)
                    # print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                    # grid.displayGrid()
                    if grid.checkWin():
                        # print ("RED WINS\n")
                        self.yAgent.getReward(1)
                        red_wins += 1
                        win = True
                        break
                else:
                    move = self.yAgent.getMove(BLACK,grid)
                    actions.append(move)
                    grid.makeMove(BLACK, move)
                    # print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                    # grid.displayGrid()
                    if grid.checkWin():
                        # print("BLACK WINS\n")
                        self.yAgent.getReward(-1)
                        win = True
                        yellow_wins += 1
                        break   
            if not win:
                self.yAgent.getReward(0.2)
                # print("DRAW\n")
            grid.resetGrid()   
        with open('QVals.dat', 'wb') as handle:
            pickle.dump(self.yAgent.Q, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return [red_wins,yellow_wins]
class QLearner:
    def __init__(self, env, player=1, epsilon=0.8, lr=0.1, gamma=0.9):
        self.env = env
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma
        self.player = player
        self.Q = dict()
    def get_hash(self, state, action):
        return " ".join(map(str, np.append(state.flatten(), action)))
    def get_action(self, state):
        if np.random.uniform() > self.epsilon:
            return self.env.sample_action(state)[0]

        max_q = -np.inf
        valid_actions = self.env.getPossibleActions(state)
        for action in valid_actions:
            state_action = self.get_hash(state, action)
            if state_action not in self.Q.keys():
                self.Q[state_action] = np.random.uniform() * 100
            if self.Q[state_action] > max_q:
                max_q = self.Q[state_action]
                best_action = action
        return best_action

    def update(self, transition):
        state, action, reward, next_state = transition

        state_action = self.get_hash(state, action)

        if state_action not in self.Q.keys():
            self.Q[state_action] = np.random.uniform() * 100
        for action in self.env.get_valid_actions(state):
            state_action_i = self.get_hash(state, action)
            if state_action_i not in self.Q.keys():
                self.Q[state_action_i] = np.random.uniform() * 100

        self.Q[state_action] += self.lr * (
            reward
            + self.gamma * np.max(
                [self.Q[self.get_hash(next_state,action)] for action in self.env.getPossibleActions(next_state)]
            )
            - self.Q[state_action]
        )
def play():
    
    player1= QLearner()
    red,black =0,0
    for x in range(100000):
        player1.reset()
        board = Board(4,5)
        rroot = Node(0, 0, None, board.grid, board.cols, board.moves, 0,2)
        rroot.populateNode(RED)
        rAgent = Agent(RED)
        actions=[]
        for i in range(board.MAX_MOVES):
            playerNum = i%2 +1 
            if playerNum == 1:
                move = player1.getMove(playerNum,board)
                # print(f"{move} given by Q for state")
                # board.displayGrid()
            else:
                move = rd.choice(board.getPossibleActions())

                # print(f"{move} given by random for state")
                # board.displayGrid()                
            # print(board.cols)
            # print(move)
            board.makeMove(playerNum,move)
            board.displayGrid()

            # board.displayGrid()
            if board.checkWin():
                if playerNum == 1:
                    print("RED WINS")
                    red+=1
                    player1.getReward(10)
                else:
                    print("BLACK WINS")
                    player1.getReward(-5)
                    black+=1
                break
        else:
            print("DRAW")
            player1.getReward(2)
        if x%1000 == 0:
            print(f"{x} games played")
            print(f"{red} red wins and {black} black wins")
            red,black =0,0
            print(len(player1.Q.items()),"States discovered")

def main():
    
    # print("************ Sample output of your program *******")

    game1 = [[0,0,0,0,0],
          [0,0,0,0,0],
          [0,0,1,0,0],
          [0,2,2,0,0],
          [1,1,2,2,0],
          [2,1,1,1,2],
        ]


    game2 = [[0,0,0,0,0],
          [0,0,0,0,0],
          [0,0,1,0,0],
          [1,2,2,0,0],
          [1,1,2,2,0],
          [2,1,1,1,2],
        ]

    
    game3 = [ [0,0,0,0,0],
              [0,0,0,0,0],
              [0,2,1,0,0],
              [1,2,2,0,0],
              [1,1,2,2,0],
              [2,1,1,1,2],
            ]

    # print('Player 2 (Q-learning)')
    # print('Action selected : 2')
    # print('Value of next state according to Q-learning : .7312')
    # PrintGrid(game1)


    # print('Player 1 (MCTS with 25 playouts')
    # print('Action selected : 1')
    # print('Total playouts for next state: 5')
    # print('Value of next state according to MCTS : .1231')
    # PrintGrid(game2)

    # print('Player 2 (Q-learning)')
    # print('Action selected : 2')
    # print('Value of next state : 1')
    # PrintGrid(game3)
    
    # print('Player 2 has WON. Total moves = 14.')
    
# if __name__=='__main__':
#     main()
from tqdm import tqdm
vals = []
# def MCTS200v40():
    
#     for j in tqdm(range(50)):
#         red =0
#         black=0
#         for i in range(50):
#             grid = Board(6,5)
#             rroot = Node(0, 0, None, grid.grid, grid.cols, grid.moves, 0,0.1*(j+1))
#             yroot = Node(0, 0, None, grid.grid, grid.cols, grid.moves, 0,0.1*(j+1))
#             ca4 = c4WithMCTS(yroot, rroot, grid)
#             x = ca4.play(1, 200,40)
#             red += x[0]
#             black += x[1]
#             grid.resetGrid()
#         print(red,black,50-red-black, (red+(50-red-black)/2)/(black+(50-red-black)/2),"for c = ",0.1*(j+1))
#         vals.append([red,black,50-red-black])
# red =0 
# black = 0
# for i in tqdm(range(100)):
#     grid = Board(6,5)
#     rroot = Node(0, 0, None, grid.grid, grid.cols, grid.moves, 0,2)
#     yroot = Node(0, 0, None, grid.grid, grid.cols, grid.moves, 0,2)
#     ca4 = c4WithMCTS(yroot, rroot, grid)
#     x = ca4.play(1, 200,40)
#     red += x[0]
#     black += x[1]
#     grid.resetGrid()
# print("RED WINS: %s"%(red))
# print("BLACK WINS: %s"%(black))
# print("DRAW: %s"%(50-red-black))
# play()
# TDAgent= QLearner()
# ca4 = c4WithQLearning(TDAgent)
# x = ca4.play(400000,0)
# # play()
# play()
# MCTS200v40()
# print(vals)
# play()
