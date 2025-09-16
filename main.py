import numpy as np
import matplotlib.pyplot as plt
import random
import heapq
from matplotlib.colors import ListedColormap, BoundaryNorm

class Node():
    def __init__(self, row_cord, col_cord, walkable = True):
        self.row_cord = row_cord 
        self.col_cord = col_cord
        self.walkable = walkable
        self.gcost = float('inf') #Cost from start node
        self.hcost = float('inf') #Cost of the heuristic
        self.fcost = float('inf') #Total cost
        self.parent = None

    def __lt__(self, other):
        return self.fcost < other.fcost #Priority queues comparison based on the total cost of each node IN CASE NEEDED??
    
    def __repr__(self):
        return f'Node({self.row_cord}, {self.col_cord}, {"walkable" if self.walkable else "wall"})'

class Maze():
    def __init__(self, mazeGrid):
        self.rows = len(mazeGrid)
        self.cols = len(mazeGrid[0])
        self.grid = []
        self.start = None
        self.goal = None

        for row_cord, row in enumerate(mazeGrid):
            node_row = []
            for col_cord, cell in enumerate(row):
                walkable = cell != 1
                node = Node(row_cord, col_cord, walkable)

                if cell == 'S':
                    self.start = node
                elif cell == 'G':
                    self.goal = node

                node_row.append(node)
            self.grid.append(node_row)

    def get_neighbours(self, node):
        neighbours = []
        directions = [
            (-1, 0), #UP
            (1, 0), #DOWN
            (0, -1), #LEFT
            (0, 1) #RIGTH
        ]

        for row_shift, col_shift in directions:
            new_row, new_col = node.row_cord + row_shift, node.col_cord + col_shift
            if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                new_neighbour = self.grid[new_row][new_col]
                if new_neighbour.walkable:
                    neighbours.append(new_neighbour)

        return neighbours
    
    def reconstruct_path(self, end_node):
        path = []
        current = end_node
        while current: #While there's something
            path.append((current.row_cord, current.col_cord))
            current = current.parent
        path.reverse()
        return path


class DisjointSet():
    def __init__(self):
        self.parent = {}

    def create_set(self, cell):
        self.parent[cell] = cell

    def find_cell(self, cell):
        if self.parent[cell] != cell:
            self.parent[cell] = self.find_cell(self.parent[cell])
        return self.parent[cell]

    def union(self, cell1, cell2):
        root1 = self.find_cell(cell1)
        root2 = self.find_cell(cell2)
        if root1 != root2:
            self.parent[root2] = root1


def generate_maze(rows, cols, obstacle_density): #Creates random maze
    maze = []
    for i in range(rows):
        row = []
        for j in range(cols):
            if(i == 0 and j == 0):
                row.append('S') #Start point
            elif(i == rows - 1 and j == cols - 1):
                row.append('G') #End point
            else:
                cell = 1 if random.random() < obstacle_density else 0
                row.append(cell)
        maze.append(row)
    return maze

def generate_maze_grid(rows, cols):
    height = rows*2 + 1
    width = cols*2 + 1
    grid = [[1 for _ in range(width)] for _ in range(height)]
    for i in range(rows):
        for j in range(cols):
            grid[2*i+1][2*j+1] = 0
    return grid

def list_walls(rows, cols):
    walls = []
    for row in range(rows):
        for col in range(cols):
            cell = (row, col)
            #Horizontal walls
            if col < cols -1:
                neighbour = (row, col + 1)
                wall_pos = (row*2 + 1, col*2 + 2)
                walls.append((cell, neighbour, wall_pos))

            #Vertical walls
            if row < rows - 1:
                neighbour = (row + 1, col)
                wall_pos = (row*2 + 2, col*2 + 1)
                walls.append((cell, neighbour, wall_pos))
    #Random asortment of the walls
    return walls

def generate_maze_kruskal(rows, cols):
    grid = generate_maze_grid(rows, cols)
    walls = list_walls(rows, cols)
    random.shuffle(walls)

    DisSet = DisjointSet() #set created
    for r in range(rows):
        for c in range(cols):
            DisSet.create_set((r, c))

    for cell1, cell2, wall_pos in walls:
        if DisSet.find_cell(cell1) != DisSet.find_cell(cell2):
            DisSet.union(cell1, cell2)
            wall_row, wall_col = wall_pos
        grid[wall_row][wall_col] = 0 #Wall removed

    #Final touches
    grid[1][1] = 'S'
    grid[rows*2 - 1][cols*2 - 1] = 'G'
    return grid

#A Star implementation

def heuristic(node_a, node_b):
    return abs(node_a.row_cord - node_b.row_cord) + abs(node_a.col_cord - node_b.col_cord) #Manhattan distance

def aStar(maze):
    start = maze.start
    goal = maze.goal
    open_list = [] #This is a priority queue (lowest fcost is at the front so it is popped first) all nodes to consider are pushed
    heapq.heappush(open_list, (0, start))
    start.gcost = 0
    start.hcost = heuristic(start, goal)
    start.fcost = start.gcost + start.hcost

    closed_set = set() #All popped nodes are stored here so that no node is repeated, when pushed into list it is first searched in case it is here

    while open_list:
        _, current = heapq.heappop(open_list) #Node to consider

        if current == goal:
            return maze.reconstruct_path(goal) #Path found

        closed_set.add((current.row_cord, current.col_cord)) #Node added to closed set

        for neighbour in maze.get_neighbours(current):
            if (neighbour.row_cord, neighbour.col_cord) in closed_set:
                continue #If already visited continue

            new_node_g = current.gcost + 1 #Each step has a cost of 1, so neighbours will have a cost of 1 with respect to the current
            #The first time we visit a neighbor, its g will be infinity, so this condition will definitely be true — that's how A* discovers new paths
            if new_node_g < neighbour.gcost: #We only want to update a neighbor if this new path is better (shorter) than any previously known path to it
                neighbour.gcost = new_node_g #We update the neighbour's costs
                neighbour.hcost = heuristic(neighbour, goal)
                neighbour.fcost = neighbour.gcost + neighbour.hcost
                neighbour.parent = current #We link the new node to our current node (we move a step) / Used for path reconstruction later
                heapq.heappush(open_list, (neighbour.fcost, neighbour)) #We add the neighbour to the list of nodes to consider

    return None #If no more nodes are left to be considered, all will have moved to the closed set and there is no solution

def create_solution_grid(maze_grid, path):
    solution_grid = []
    path_set = set(path)

    for i, row in enumerate(maze_grid):
        new_row = []
        for j, cell in enumerate(row):
            if (i, j) in path_set and cell == 0:
                new_row.append('X')
            else:
                new_row.append(cell)
        solution_grid.append(new_row)

    return solution_grid

def display_maze(maze):
    for row in maze:
        print(f' '.join(str(cell) for cell in row))

def plot_maze(grid):
    colour_map = { #Definition of our palette
        0:0, #White
        1:1, #Black
        'S':2, #Green
        'G':3, #Red
        'X':4 #Yellow
    }

    palette = ListedColormap(['white', 'black', 'green', 'red', 'yellow'])
    visual_grid = np.zeros((len(grid), len(grid[0])))

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            visual_grid[i][j] = colour_map[grid[i][j]]

    plt.imshow(visual_grid, cmap=palette) #Implementation of our palette as colour map
    plt.xticks([])
    plt.yticks([])
    plt.title('Maze')
    plt.show()


if __name__ == '__main__':
    raw_grid = generate_maze_kruskal(100,100)
    maze = Maze(raw_grid)
    solution = aStar(maze)
    #print(solution)
    print(f'Solution: {len(solution)} steps')
    solved_maze = create_solution_grid(raw_grid, solution) 
    #display_maze(raw_grid)
    #display_maze(solved_maze)
    plot_maze(raw_grid)
    plot_maze(solved_maze)

# Program by Pedro Oubiña S. 2025
