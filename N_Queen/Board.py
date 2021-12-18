import random
class Board:
    def __init__(self,n,board = None):
        """
        Parameters
        ----------
        n : int
            the side length of the board.
        board : list(int), optional
            Board description. The default is None.

        Returns
        -------
        None.
        """
        self.n = n
        if board !=None:
            assert (n==len(board))
            self.board = board
        else:
            self.board = [0]*n
            for i in range(n):
                self.board[i] = random.randint(0,n-1)
    def printBoard(self):
        """
        Used to print the Board.

        Returns
        -------
        None.
        """
        for i in range(self.n):
            for j in range(self.n):
                if j<self.board[i] or j>self.board[i]:
                    print(" _ |",end="")
                else :
                    print(" Q |",end="")
            print("")
            for j in range(self.n):
                print("----",end="")
            print("")
    
    def getState(self):
        return self.board.copy()
    def setState(self,board):
        self.board = board