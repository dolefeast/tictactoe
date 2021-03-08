import numpy as np
import time
import pickle

rows = cols = 3

class State:
    def __init__(self, p1, p2): #p1 uses symbol 1, p2 uses symbol 2. Vacancy is 0
        self.board = np.zeros((rows, cols))
        self.p1 = p1 #p1 plays first
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None #Hashed board
        self.playerSymbol = 1

    #Unique hash of current board state
    def getHash(self):
        self.boardHash = str(self.board.reshape(rows*cols))
        return self.boardHash 

    def winner(self):
        for i in range(rows):
            #Columns
            if sum(self.board[:, i]) == -3:
                self.isEnd = True
                return -1

            if sum(self.board[:, i]) == 3:
                self.isEnd = True
                return 1
            #Rows
            if sum(self.board[i, :]) == -3:
                self.isEnd = True
                return -1

            if sum(self.board[i, :]) == 3:
                self.isEnd = True
                return 1

        if sum(np.diag(np.fliplr(self.board))) == 3:
            self.isEnd = True
            return 1

        if sum(np.diag(np.fliplr(self.board))) == -3:
            self.isEnd = True
            return -1

        if sum(np.diag(self.board)) == 3:
            self.isEnd = True
            return 1

        if sum(np.diag(self.board)) == -3:
            self.isEnd = True
            return -1

        #Tie
        if len(self.availablePositions()) == 0:
            self.isEnd = True
         #   print('Tie!')
            return 0

        self.isEnd = False
        return None

    def availablePositions(self):
        positions = []
        for col in range(cols):
            for row in range(rows):
                if self.board[row, col] == 0:
                    positions.append([row, col])

        return positions

    def updateState(self, position):
        self.board[position[0], position[1]] = self.playerSymbol
        self.playerSymbol *= -1 #Switch players

    def giveReward(self, result): #Backpropagate results
        #result = self.winner()
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.1)
            self.p2.feedReward(0.5)
    
    def reset(self):
        self.board = np.zeros((rows, cols))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1

    def full_reset(self):
        self.p1.reset()
        self.p2.reset()
        self.reset()

    def play(self, rounds=100):
        for i in range(rounds):
            wincount = 0
            if i%1000==0:
                print('Rounds {}'.format(i))
            #print('New game!')
            while not self.isEnd:
                #input()
                #Player 1 plays
                positions = self.availablePositions()
                p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
                #print('p1 moves')
                self.updateState(p1_action)
                board_hash = self.getHash()
                self.p1.addState(board_hash)

                win = self.winner()
                #self.showBoard()
                if win is not None:
                    #if win == 1:
                    #    print(self.p1.name, 'wins!')
                    self.giveReward(win)
                    self.full_reset()
                    break

                else:
                    #input()
                    positions = self.availablePositions()
                    p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                    #print('p2 moves')
                    self.updateState(p2_action)
                    #self.showBoard()
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)

                    win = self.winner()
                    if win is not None:
                        #if win == -1:
                        #    #print(self.p2.name, 'wins!')
                        self.giveReward(win)
                        self.full_reset()
                        break
            wincount += win
            print(wincount)
                    
    def play2(self):
        while not self.isEnd:
                #Player 1 plays
                positions = self.availablePositions()
                p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
                self.updateState(p1_action)
                board_hash = self.getHash()
                self.p1.addState(board_hash)
                #print(board_hash)

                win = self.winner()
                self.showBoard()
                if win is not None:
                    self.giveReward(win)
                    self.full_reset()
                    break

                else:
                    positions = self.availablePositions()
                    p2_action = self.p2.chooseAction(positions)

                    self.updateState(p2_action)
                    self.showBoard()

                    win = self.winner()
                    if win is not None:
                        if win == -1:
                            print(self.p2.name, 'wins!')
                        else:
                            print('Tie!')
                        self.reset()
                        break

    def showBoard(self):
        for i in range(rows):
            print('-------------')
            out = '| '
            for j in range(cols):
                if self.board[i, j] == 1: 
                    token = 'X'
                if self.board[i, j] == -1: 
                    token = 'O'
                if self.board[i, j] == 0: 
                    token = ' '

                out += token + ' | '

            print(out)
        print('-------------')

class Player:
    def __init__(self, name, exp_rate=0.3): 
        #exp_rate is the probability the agent doesn't explore and plays serious
        self.name = name
        self.states = [] #records positions 
        self.lr = 0.2 #What's this?
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9 #What's this?
        self.states_value = {}

    def getHash(self, board):
        boardHash = str(board.reshape(cols*rows))
        return boardHash

    def chooseAction(self, positions, current_board, symbol):
        if np.random.uniform(0, 1) <= self.exp_rate:#Explore, take random action
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else:
            value_max = -999
            for p in positions: #Evaluate possible moves
                next_board = current_board.copy() #Save copy of current board
                next_board[p] = symbol #Update copy
                next_boardHash = self.getHash(next_board) #Hash of board
                if self.states_value.get(next_boardHash) is None: #Empty dict
                    value = 0
                else:
                    value = self.states_value.get(next_boardHash) #Save board

                if value >= value_max: #Take the highest value action
                    value_max = value
                    action = p
            #print("{} takes action {}".format(self.name, action))

        return action

    #Append hash state
    def addState(self, state):
        self.states.append(state)

    #Backpropagate at end of game
    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr*(self.decay_gamma * reward - self.states_value[st]) #Guess gamma is a learning constant
            reward = self.states_value[st]

    def reset(self):
        self.states = []
        
    def savePolicy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()


class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions):
        while True:
            row = int(input('Input action row:'))
            col = int(input('Input action col:'))
            action = [row, col]

            if action in positions:
                return action

    def addState(self, state):
        pass

    
    def feedReward(self, reward):
        pass

    def reset(self):
        pass


p1 = Player('p1')
p2 = Player('p2', exp_rate=1)

st = State(p1, p2)

st.play(10000)


p3 = HumanPlayer('Santi')
st2 = State(p1, p3)

st2.play2()



