import numpy as np
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
        self.boardHash = str(self.board.reshape(self.rows*self.cols))
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
            return 0

        self.isEnd = False
        return None

    def availablePositions(self):
        positions = []
        for col in range(self.cols):
            for row in range(self.rows):
                if self.board[row, col] == 0:
                    positions.append([row, col])

        return positions

    def updateState(self, position):
        self.board[position] = self.playerSymbol
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
        self.board = np.zeros((self.rows, self.cols))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1

    def full_reset(self):
        self.p1.reset()
        self.p2.reset()
        self.reset()

    def play(self, rounds=100):
        for i in range(rounds):
            if i%1000==0:
                print('Rounds {}'.format(i))
            while not self.isEnd:
                #Player 1 plays
                positions = self.availablePositions()
                p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
                self.updateState(p1_action)
                board_hash = self.getHash()
                self.p1.addState(board_hash)

                win = self.winner()
                if win is not None:
                    self.giveReward(self, win)
                    self.full_reset()
                    break

                else:
                    positions = self.availablePositions()
                    p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                    self.updateState(p2_action)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)

                    win = self.winner()
                    if win is not None:
                        self.giveReward(self, win)
                        self.full_reset()
                        break
                    
    def play2(self):
        while not self.isEnd:
                #Player 1 plays
                positions = self.availablePositions()
                p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
                self.updateState(p1_action)
                board_hash = self.getHash()
                self.p1.addState(board_hash)

                win = self.winner()
                if win is not None:
                    self.giveReward(self, win)
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
                            print(self.ps.name, 'wins!')
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
        #exp_rate is the probability the agent doesn't explore and plays serious.
        self.name = name
        self.states = [] #records positions 
        self.lr = 0.2 #What's this?
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9 #What's this?
        self.states_value = {}

