# Imports
import numpy as np
import random as rd
from copy import deepcopy
from collections import defaultdict
import gzip
import dill as pickle
#Vars
INF = 100000000
RED = 2
BLACK = 1
DRAW = -1
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
        cnt = 0
        while True:
            if not self.inBoundary(x, y) or color != grid[x][y]:
                return False
            x+=1
            cnt+=1
            if cnt==4:
                return True

    def checkForWinningMove(self, color):
        temp = self.grid.copy()
        for i in range(7):
            if self.cols[i] >= 0:
                temp[self.cols[i]][i] = color
                if self.checkWinVirtual(temp, self.cols[i], i):
                    return i
                temp[self.cols[i]][i] = 0
        return -1

    def checkHorizontal(self, grid, lastMove):
        x = lastMove[0]
        y = lastMove[1]
        color = grid[x][y]
        cnt = 0
        while True:
            if not self.inBoundary(x, y) or color != grid[x][y]:
                break
            y += 1
            cnt += 1
            if cnt == 4:
                return True

        x = lastMove[0]
        y = lastMove[1] - 1

        while True:
            if not self.inBoundary(x, y) or color != grid[x][y]:
                break
            y -= 1
            cnt += 1
            if cnt == 4:
                return True

        if cnt >= 4:
            return True
        return False

    def checkDiagonal(self, grid, lastMove):
        x = lastMove[0]
        y = lastMove[1]
        color = grid[x][y]
        cnt = 0
        while True:
            if not self.inBoundary(x, y) or color != grid[x][y]:
                break
            x -= 1
            y += 1
            cnt += 1
            if cnt == 4:
                return True
        x = lastMove[0] + 1
        y = lastMove[1] - 1
        while True:
            if not self.inBoundary(x, y) or color != grid[x][y]:
                break
            x += 1
            y -= 1
            cnt += 1
            if cnt == 4:
                return True
        x = lastMove[0]
        y = lastMove[1]
        cnt = 0
        while True:
            if not self.inBoundary(x, y) or color != grid[x][y]:
                break
            x -= 1
            y -= 1
            cnt += 1
            if cnt == 4:
                return True
        x = lastMove[0] + 1
        y = lastMove[1] + 1
        while True:
            if not self.inBoundary(x, y) or color != grid[x][y]:
                break
            x += 1
            y += 1
            cnt += 1
            if cnt == 4:
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
    def __init__(self, total_score, n, parent, state, cols, moveCnt, level,c):
        self.totalScore = total_score
        self.c = c
        self.n = n
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

        maxIndex,maxValue = 0 , -1*INF

        for i in range(len(self.children)):
            if ucbs[i] != None and ucbs[i] > maxValue:
                maxIndex = i
                maxValue = ucbs[i]

        maxNode = self.children[maxIndex]
        return maxNode, maxIndex, ucbs

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
        return 3-color


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

    def getBestMove(self, actions, Iters, root, grid):
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
            if not node: 
                prev_node.populateNode(color)
                node = prev_node.children[action]

        if node.checkLeaf():
            node.populateNode(self.color)

        curr = node
        change = False

        while count < Iters:
            if not change:
                curr = node
            if curr.checkLeaf():
                if curr.n == 0:
                    if curr.isTerminal:
                        reward = self.getRewardTerminal(curr.winColor)
                        change = False
                        count += 1
                        curr.backpropagate(reward)                        
                        continue
                    else:
                        vgrid = curr.state.copy()
                        vcols = curr.cols.copy()
                        colorToMove = BLACK if curr.moveCnt%2 == 1 else RED
                        reward = self.rollout(vgrid, vcols, curr.moveCnt, colorToMove)
                        change = False
                        count += 1
                        curr.backpropagate(reward)
                        continue
                else:
                    colorToMove = BLACK if curr.moveCnt%2 == 1 else RED

                    if curr.isTerminal:
                        reward = self.getRewardTerminal(curr.winColor)
                        change = False
                        count += 1
                        curr.backpropagate(reward)
                        continue
                    curr.populateNode(colorToMove)
                    curr, _, _ = curr.getMaxUcbNode(root.n)
                    if curr.isTerminal:
                        reward = self.getRewardTerminal(curr.winColor)
                        change = False
                        curr.backpropagate(reward)
                        count += 1
                        continue
                    vgrid = curr.state.copy()
                    vcols = curr.cols.copy()
                    colorToMove = BLACK if curr.moveCnt%2 == 1 else RED
                    reward = self.rollout(vgrid, vcols, curr.moveCnt, colorToMove)
                    curr.backpropagate(reward)
                    
                    count += 1
                    change = False
                    continue

            else:
                change = True
                curr, _ , _= curr.getMaxUcbNode(root.n)
        _, action, _ = node.getMaxUcbNode(root.n)
        return root, action
class connect4WithMCTS:
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
            actions = []
            for j in range(self.grid.MAX_MOVES):

                if j%2 == 0:  #red move
                    self.rroot, action = self.rAgent.getBestMove(actions, n_iterations1, self.rroot, self.grid)
                    actions.append(action)
                    self.grid.makeMove(RED, action)
                    print("-----------------")
                    self.grid.displayGrid()
                    if self.grid.checkWin():
                        print("Player1 WINS\n")
                        red_wins += 1
                        win = True
                        break
                else: #BLACK move
                    self.yroot, action = self.yAgent.getBestMove(actions, n_iterations2, self.yroot, self.grid)
                    actions.append(action)
                    self.grid.makeMove(BLACK, action)
                    print("-------------------")
                    self.grid.displayGrid()
                    if self.grid.checkWin():
                        print("Player2 WINS\n")
                        win = True
                        yellow_wins += 1
                        break   
            if not win:
                print("DRAW\n")
            self.grid.resetGrid()   
        return [red_wins,yellow_wins]
class QLearner:
    def __init__(self):
        self.Q =defaultdict(lambda: 0.2)
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.05
        self.lastAfterstate = None
    def getMove(self,playerNumber,board:Board):
        actions = board.getPossibleActions()
        afterStates = board.getAfterStates(playerNumber)
        afterStateValues = [self.Q[afterState.hashValue] for afterState in afterStates]
        best_afterstate_val, best_index = max(zip(afterStateValues, range(len(afterStateValues))))
        if rd.random() < self.epsilon:
            move, afterstate = rd.choice(list(zip(actions, afterStates)))
        else:
            move = actions[best_index]
            afterstate = afterStates[best_index]
        if self.lastAfterstate is not None:
            self.Q[self.lastAfterstate.hashValue] += self.alpha *self.gamma* best_afterstate_val
        self.lastAfterstate = afterstate
        return move
    def getReward(self,reward):
        self.Q[self.lastAfterstate.hashValue] += self.alpha * (reward - self.Q[self.lastAfterstate.hashValue])
    def reset(self):
        self.lastAfterstate = None
    def toGreedyOrNotToGreedy(self,greedy):
        if greedy:
            self.epsilon = 0
            self.alpha = 0
        else:
            self.epsilon=0.05
            self.alpha = 0.1
def QLvsRandom():
    
    player1= QLearner()
    with open('2019A7PS0160G_VISHAL.dat', 'wb') as src, gzip.open('2019A7PS0160G_VISHAL.dat.gz', 'rb') as dst:
        src.writelines(dst)
    with open('./2019A7PS0160G_VISHAL.dat', 'rb') as pickle_file:
        player1.Q = pickle.load(pickle_file)
    # print(player1.Q)
    red,black =0,0
    player1.toGreedyOrNotToGreedy(True)
    for x in range(1):
        player1.reset()
        board = Board(4,5)
        rroot = Node(0, 0, None, board.grid, board.cols, board.moves, 0,2)
        rroot.populateNode(RED)
        rAgent = Agent(RED)
        for i in range(board.MAX_MOVES):
            playerNum = i%2 +1 
            if playerNum == 1:
                move = player1.getMove(playerNum,board)
                # print(f"{move} given by Q for state")
            else:
                move = rd.choice(board.getPossibleActions())

                # print(f"{move} given by random for state")
            board.makeMove(playerNum,move)
            print('------------------------')
            board.displayGrid()

            # board.displayGrid()
            if board.checkWin():
                if playerNum == 1:
                    # print("RED WINS")
                    red+=1
                else:
                    # print("BLACK WINS")
                    black+=1
                break
    print(red,black)
def MCTS200v40():
    
    
    red =0
    black=0
    grid = Board(6,5)
    rroot = Node(0, 0, None, grid.grid, grid.cols, grid.moves, 0,2)
    yroot = Node(0, 0, None, grid.grid, grid.cols, grid.moves, 0,2)
    ca4 = connect4WithMCTS(yroot, rroot, grid)
    x = ca4.play(1, 200,40)
    red += x[0]
    black += x[1]
    grid.resetGrid()
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
    MCTS200v40()

    QLvsRandom()
    print("In QL , 1 is player 1 , 2 is player 2\nwhereas in MCTS 2 is player 1  and 1 is player 2")

if __name__=='__main__':
    main()
