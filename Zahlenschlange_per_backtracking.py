
"""
Zahlenschlange-Rätsel mithilfe des Backtracking-Algorithmus
"""


import random
from PIL import Image, ImageDraw


board = """
1 27 29 27 40 36 20 39 2 2
29 22 6 29 19 15 30 40 25 21
7 40 17 2 34 8 24 33 20 27
19 17 2 17 39 35 40 15 29 29
8 25 39 7 37 27 21 5 37 9
4 23 26 9 38 41 11 18 2 10
3 32 18 28 14 33 24 23 6 3
14 42 3 34 8 38 20 21 42 13
33 15 20 30 6 12 31 20 35 32
29 16 11 15 16 4 26 36 6 43
"""

START = (0,0)
END = (9,9)
PATH_LEN = 43


def board_to_list(board):
    l = []
    for line in board.strip().split('\n'):
        l.append([int(n.strip()) for n in line.strip().split(' ')])
    return l


def get_candidates(square):
    i,j = square
    return [(i,j) for (i,j) in [(i, j+1), (i+1, j), (i, j-1), (i-1, j)]
            if 0 <= i <= 9 and 0 <= j <= 9]
    
    
def ok(board, path):
    # if the path is yet empty
    if not path:
        return True
    
    #all squares in the path must be unique (i.e. no backing up, no path crossing)
    if len(set(path)) != len(path):
        return False
    
    # all numbers in the path must be unique
    numbers = [board[i][j] for (i,j) in path]
    if len(set(numbers)) != len(numbers):
        return False
    
    # the path length must be less than or equal to 43
    if len(path) > 43:
        return False
    
    # the first square must be (0,0)
    if path[0] != START:
        return False
    
    # the last square must be (9,9)
    if len(path) == PATH_LEN and path[-1] != END:
        return False
    
    # next square must be a valid move from the previous square
    if len(path) >= 2:
        for n, square in enumerate(path[:-1]):
            candidates = get_candidates(square)
            random.shuffle(candidates)
            if path[n+1] not in candidates:
                return False
    return True


def backtrack(board, path=[]):
    # Base Case
    if len(path) == PATH_LEN:
        return path
    
    # Recursive Case
    if not path: path = [START,]
    for square in get_candidates(path[-1]):
        path.append(square)
        if ok(board, path):
            result = backtrack(board, path)
            if result:
                return result
        path.pop()
    return None
    
######################################################################

board = board_to_list(board)
print(*board, sep='\n', end="\n\n")

path = backtrack(board)
print(*zip(path, (board[i][j] for i,j in path)), sep='\n')



# Draw (not adeqaute)
N = 10   # number of squares
SIDE = 400

#create a blank canvas
img = Image.new(mode="RGBA", size=(SIDE, SIDE), color="black")
draw = ImageDraw.Draw(img)

square_side = round(SIDE / N)

for i,y in enumerate(range(0, SIDE, square_side)):
    for j,x in enumerate(range(0, SIDE, square_side)):
        color = "red" if (i,j) in path else "lightblue"
        draw.rectangle(xy=[(x, y), (square_side + x, square_side+y)], 
                       fill=color, outline='gray', width=1)
        draw.text(xy=(x + square_side//2 - 3, y + square_side//2 - 3), 
                  text=str(board[i][j]), fill="black", align='center')

#save and show the image
img.save("Zahlenschlange.png")
img.show("Zahlenschlange.png")
