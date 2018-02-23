""" Simple implementation of GridWorld
Author: Jin Yeom (jinyeom@utexas.edu) 
"""

import numpy as np
from copy import deepcopy

# Valid grid types.
GRID = 0 # can go this way
WALL = 1 # can't go this way
GOAL = 2 # good terminal state 
FIRE = 3 # bad terminal state

# Valid move directions.
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridWorld(object):
    """ 2D GridWorld """
    
    def __init__(self, m):
        if not isinstance(m, np.ndarray) or len(m.shape) != 2:
            raise TypeError("GridWorld map must be a 2D numpy.ndarray")
        r, c = m.shape
        self._m = m
        self._V = np.zeros((r, c)) # Values for each state
        self._Q = np.zeros((r, c, 4)) # Q values for 4 directions in each state
       
    def __getitem__(self, i, j):
        return self._m[i][j]
    
    @property
    def M(self):
        return deepcopy(self._m)
    
    @property
    def V(self):
        return deepcopy(self._V)
    
    @property
    def Q(self):
        return deepcopy(self._Q)
    
    def get_V(i, j):
        """ Return the value of state (i, j) """
        return self._V[i][j]

    def get_Q(i, j, a):
        """ Return the Q-value of state (i, j) for action a """
        return self._Q[i][j][a]
    
    def set_V(i, j, v):
        self._V[i][j] = v
    
    def set_Q(i, j, a, q):
        self._Q[i][j][a] = q
        self._V[i][j] = np.max(self._Q[i][j])
          
    def R(self, i, j):
        """ Reward function """
        return [-1, -1, 500, -500][self._m[i][j]]
    
    def T(self, i, j, a):
        """ Transition function """
        if a not in [UP, RIGHT, DOWN, LEFT]:
            raise ValueError("invalid direction")
        next_i, next_j = i, j
        if a == UP:
            next_i -= 1
        elif a == RIGHT:
            next_j += 1
        elif a == DOWN:
            next_i += 1
        elif a == LEFT:
            next_j -= 1
        # if the next position is invalid, revert back.
        if next_i >= r or next_j >= c or self._m[next_i][next_j] == WALL:
            next_i, next_j = i, j
        return next_i, next_j
        
    def show(self):
        """ Print the map """
        def sym(i, j):
            return [" ", "#", "o", "x"][self._m[i][j]]
        r, c = self._m.shape
        print("-".join(["+" for j in range(c + 1)]))
        for i in range(r):
            print("|%s|" % "|".join(["%s" % sym(i, j) for j in range(c)]))
            print("-".join(["+" for j in range(c + 1)]))
                