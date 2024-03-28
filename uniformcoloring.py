from utils import *
from search import *
#from aima import Problem, Node #  Import aimacode from the aima-python library
from enum import Enum
import time

MOV_COST = 1

class Colors(Enum):
    BLUE = 1
    YELLOW = 2
    GREEN = 3

    def __str__(self):
        # return lower case name
        return '[ Color : ' + self.name.lower() + ' ]'

class Directions(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

    def __str__(self):
        return '[ Direction : ' + self.name.lower() + ' ]'

class Heuristic(Enum):
    heuristic_color_use_most_present = 1

class State():
    def __init__(self, grid, i, j):
        self.grid = grid
        self.i = i # row
        self.j = j # column
        self.id = str(i)+str(j)
        for row in grid:
            for tile in row:
                self.id = self.id+str(tile)
    
    def __lt__(self, state):
        return False

class UniformColoring(Problem):
    """The abstract class for a formal problem.  You should subclass this and
    implement the method successor, and possibly __init__, goal_test, and
    path_cost. Then you will create instances of your subclass and solve them
    with the various search functions."""

    def __init__(self, initial, heuristic_type):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.heuristic_type = heuristic_type
    
    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are many
        actions, consider yielding them one at a time in an iterator, rather
        than building them all at once."""
        actions=[]
        if (state.i != self.initial.i) or (state.j != self.initial.j):
            for color in Colors:  # action color tile
                if (color.value != state.grid[state.i][state.j]):
                    actions.append(color)
        for direction in Directions:  # action move
            coords=(state.i+direction.value[0],state.j+direction.value[1])
            if coords[0] in range(state.grid.shape[0]) and coords[1] in range(state.grid.shape[1]):
                actions.append(direction)
        return actions
    
    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        if action in Directions:
            return State(state.grid,state.i+action.value[0],state.j+action.value[1])
        else:
            grid=np.copy(state.grid)
            grid[state.i][state.j]=action.value
            return State(grid,state.i,state.j)

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal, as specified in the constructor. Implement this
        method if checking against a single self.goal is not enough."""
        test_color = None
        if ((state.i, state.j) == (self.initial.i, self.initial.j)):  # if it's back at the start position
            for color in Colors:
                if color.value in state.grid:
                    if test_color == None:
                        test_color = color.value
                    if color.value != test_color:
                        return False
        else:
            return False
        return True

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        if action in Colors:
            return c + action.value
        return c + MOV_COST
    
    def color_initial_choice(self, node):
        # count the number of each color in the grid
        colors = {
            Colors.BLUE.value: 0, 
            Colors.YELLOW.value: 0, 
            Colors.GREEN.value: 0
        }
        for row in node.state.grid:
            for tile in row:
                if tile == 0: #0 corresponds to the initial position T
                    continue
                colors[tile] += 1

        num_tot_grid = (node.state.grid.shape[0] * node.state.grid.shape[1]) -1
        res = []
        for color in colors:
            res.append((num_tot_grid - colors[color])*2)
        
        return res.index(min(res)) + 1
    
    def manhattan_distance(self, coord1, coord2):
        return abs(coord2[0] - coord1[0]) + abs(coord2[1] - coord1[1])
    
    def heuristic(self, node, color_choice):
        if self.heuristic_type == Heuristic.heuristic_color_use_most_present:
            return self.heuristic_color_use_most_present(node, color_choice)
    
    def heuristic_color_use_most_present(self, node, color_choice):
        """Return the heuristic value for a given state. Default heuristic
        function is 0."""
        h = 0
        grid = node.state.grid
        not_colored = []
        starting_point = []
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if(grid[i][j] != color_choice and grid[i][j] != 0):
                    not_colored.append((i, j))
                    h += color_choice
                if(grid[i][j] == 0):
                    starting_point.append((i, j))
        i = node.state.i
        j = node.state.j

        for to_color in not_colored:
            h += self.manhattan_distance((i, j), to_color)
        
        return h
    
    def h(self, node):
        """Return the heuristic value for a given state."""
        i,j=(node.state.i,node.state.j)
        color_choice = self.color_initial_choice(node)
        if node.action != None and node.action in Colors:
            return self.heuristic(node, color_choice)
        parent=node.parent
        color=None
        #If the action is in Directions the heuristic evaluation is based on the parent color
        if parent != None:
            color=parent.state.grid[parent.state.i][parent.state.j]
        if (i,j) == (self.initial.i,self.initial.j) or color==0: #If I don't have a parent color, i choose the one that minimizes the Heuristic value
            h = None
            for color in Colors:
                temp_h = self.heuristic(node, color.value)
                if h == None:
                    h=temp_h
                if temp_h < h:
                    h = temp_h
            #print("Heuristic 0,0 value:", h)
            return h
        else:
            h = self.heuristic(node, color)
            #print("Heuristic value:", h, node.state.grid, i,j)
            return h


    def value(self):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError
    
def best_first_graph_search(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    #print("#COORDS0:", node.state.i, node.state.j, node.state.grid)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    lookup_frontier=set()
    lookup_frontier.add(id(node))
    explored = set()
    while frontier:
        #print("frontiera:", len(frontier))
        node = frontier.pop()
        #print("#NODE:", node.state.i, node.state.j, node.state.grid)
        lookup_frontier.remove(id(node))
        if problem.goal_test(node.state):
            return node, 1
        elif len(explored) > 60000:
            return node, -1
        explored.add(node.state.id)
        for child in node.expand(problem):
            #print("#CHILD:", node.state.i, node.state.j, node.state.grid)
            if child.state.id not in explored and id(child) not in lookup_frontier:
                lookup_frontier.add(id(child))
                frontier.append(child)
            elif id(child) in lookup_frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    lookup_frontier.add(id(child))
                    frontier.append(child)
    return None, -1

def is_cycle(node):
    current = node
    while current.parent != None:
        if current.parent.state.id == current.state.id:
            return True
        current = current.parent
    return False

def depth_limit_search(problem, limit):
    frontier = [Node(problem.initial)]
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        elif limit == 25:
            return node
        if node.depth < limit:
            explored.add(node.state.id)
            for child in node.expand(problem):
                if child.state.id not in explored:
                    frontier.append(child)
        elif is_cycle(node):
            return 'cutoff' + str(node.depth)
    return None

def iterative_deepening_search(problem):
    for depth in range(sys.maxsize):
        result = depth_limit_search(problem, depth)
        if result != None and depth < 25:
            return result, 1
        elif result != None and depth == 25:
            return result, -1
    return None, -1

def astar_search(problem, h=None): 
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    end, bfgs_succ = best_first_graph_search(problem, lambda n: n.path_cost + h(n))
    return end, bfgs_succ

def greedy_search(problem, h=None):
    """
    Greedy best-first search is accomplished by specifying f(n) = h(n).
    """
    h = memoize(h or problem.h, 'h')
    end, bfgs_succ = best_first_graph_search(problem, lambda n: h(n))
    return end, bfgs_succ

def initialize_state(grid):
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i][j]==0:
                return State(grid,i,j)
    return None