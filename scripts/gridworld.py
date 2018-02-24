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
        self._m = m
        self._r, self._c = m.shape
        self._V = np.zeros((self._r, self._c))
        self._Q = np.zeros((self._r, self._c, 4))
       
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
    
    @property
    def S(self):
        """ Return a list of all possible states. """
        r, c = self._m.shape
        return [(i, j) for i in range(r) for j in range(c)]
    
    @property
    def shape(self):
        return self._r, self._c
    
    def set_V(self, s, v):
        self._V[s] = v
    
    def set_Q(self, s, a, q):
        self._Q[s][a] = q
        self._V[s] = np.max(self._Q[s])
          
    def R(self, s):
        """ Reward function """
        return [-1, -1, 50, -50][self._m[s]]
    
    def T(self, s, a):
        """ Transition function """
        if a not in [UP, RIGHT, DOWN, LEFT]:
            raise ValueError("invalid direction")
        next_i, next_j = s
        if a == UP:
            next_i -= 1
        elif a == RIGHT:
            next_j += 1
        elif a == DOWN:
            next_i += 1
        elif a == LEFT:
            next_j -= 1
        # if the next position is invalid, revert back.
        if next_i >= self._r or next_j >= self._c or self._m[next_i][next_j] == WALL:
            next_i, next_j = s
        return next_i, next_j
    
    def done(self, s):
        """ Check if the argument state is a terminal state. """
        if self._m[s] == GOAL or self._m[s] == FIRE:
            return True
        return False
        
    def show(self):
        """ Print the map """
        def sym(i, j):
            return [" ", "#", "o", "x"][self._m[i][j]]
        print("-".join(["+" for j in range(self._c + 1)]))
        for i in range(self._r):
            print("|%s|" % "|".join(["%s" % sym(i, j) for j in range(self._c)]))
            print("-".join(["+" for j in range(self._c + 1)]))
                