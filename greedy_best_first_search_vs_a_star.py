
"""
Greedy Best First Search
and
A* search
"""


from dataclasses import dataclass
from math import inf

@dataclass
class Node:
    state: tuple
    parent: 'Node'
    action: str or object
    reach: int   # how many steps it took the agent to reach this node from the start
    
    def __eq__(self, other):
        return tuple(self.state) == (tuple(other) if hasattr(other, '__iter__') else tuple(other.state))
    

class Maze:
    PATH = -1
    OBSTACLE = -2
    
    def __init__(self, maze: str, heuristic=None):
        self.huristic = heuristic or 'gbfs'
        self._parse_maze(maze)
    
    def _parse_maze(self, maze):
        """
        parse the maze in form of a str into a list of lists
        with 0 = path    1 = obstacle
        """
        self.maze = []
        for i,line in enumerate(maze.splitlines()):
            row = []
            for j,c in enumerate(line):
                if c.upper() in ('A', 'S', 'I'):
                    self.start = (i,j)
                    row.append(self.PATH)
                elif c.upper() in ('B', 'G', 'E', 'F'):
                    self.goal = (i,j)
                    row.append(self.PATH)
                elif c == ' ':
                    row.append(self.PATH)
                else:
                    row.append(self.OBSTACLE)
            self.maze.append(row)
        # Height and length of the maze
        self.m, self.n = i+1, j+1
        
    def get_neighbors(self, state):
        r,c = state
        
        if not(0 <= r < self.m and 0 <= c < self.n and self.maze[r][c] == self.PATH):
            raise ValueError("bad index")
        
        candidates = [('up',    (r-1,c )), 
                      ('down',  (r+1, c)), 
                      ('left',  (r, c-1)), 
                      ('right', (r, c+1))]
        neighbors = []
        for action, (r,c) in candidates:
            if 0 <= r < self.m and 0 <= c < self.n and self.maze[r][c] == self.PATH:
                neighbors.append(Node(state=(r,c), action=action, parent=None, reach=self.k))
        return neighbors
        
    def h(self, node):
        """heuristic function"""
        manhatten_dist = abs(self.goal[0] - node.state[0]) + abs(self.goal[1] - node.state[1])
        if str(self.huristic).lower() == 'gbfs':   # Greedy Best First Search
            return manhatten_dist
        elif 'a' in str(self.huristic).lower():    # a*  or a-star etc
            return manhatten_dist + node.reach
        else:
            raise ValueError("heuristic type not recognized")
    
    def solve(self):
        self.k = 0  # counter of steps made
        frontier = [Node(state=self.start, parent=None, action=None, reach=inf),]   
        explored = [] 
        
        while frontier:
            current_node = min((node for node in frontier), key=lambda node: self.h(node))
            # Check if goal
            if current_node == self.goal:
                print(f"goal at {current_node.state} reached in {self.k} moves")
                self.goal_node = current_node
                return current_node.state
            frontier.remove(current_node)
            explored.append(current_node)
            neighbors = self.get_neighbors(current_node.state)
            neighbors = [n for n in neighbors if n not in explored and n not in frontier]
            
            for n in neighbors:
                n.parent = current_node
                frontier.append(n)
            # For print
            i,j = current_node.state
            self.maze[i][j] = self.k
            self.k += 1
        
    def get_actions(self):
        node = self.goal_node
        actions = [node.action,]
        while node.parent:
            node = node.parent
            actions.append(node.action)
        return actions[::-1][1:]
            
    def get_path(self):
        node = self.goal_node
        path = [node.state,]
        while node.parent:
            node = node.parent
            path.append(node.state)
        return [self.start] + path[::-1][1:]
    
    def print(self):
        pad = len(str(max(sum(maze.maze, [])))) + 2
        self.maze[self.goal[0]][self.goal[1]] = -3
        d = {self.PATH: ' ', self.OBSTACLE: chr(9608), -3:'#'}
        print()
        for i,row in enumerate(self.maze):
            for j,c in enumerate(row):
                c = d.get(c, 0)*pad or str(c).center(pad)
                print(c, sep='', end='')
            print()
        print()
        
    
###################################################################################

maze = """\
### B               #########
#   ###################   # #
# ####                # # # #
# ################### # # # #
#                     # # # #
##################### # # # #
#   ##                # # # #
# # ## ### ## ######### # # #
# #    #   ## #         # # #
# # ## ################ # # #
### ##             #### # # #
### ############## ## # # # #
###             ##    # # # #
###### ######## ####### # # #
###### ####             #   #
A      ######################"""



maze = """\
##    #
## ## #
#B #  #
#### ##
     ##
A######"""



maze = """\
#######
#     #
##### #
#G    #
##### #
#  #  #
## # ##
     ##
S######"""



maze = """\
#######
      #
 #### #
    #G#
### # #
S   # #
# ### #
#     #
#######
"""



maze = Maze(maze, heuristic='a*')
solution = maze.solve()
print(solution)

actions = maze.get_actions()
print(actions)

path = maze.get_path()
print(path)

maze.print()

