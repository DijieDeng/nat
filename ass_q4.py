# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 14:56:40 2021

@author: 56325
"""
from random import random, randint, seed
from statistics import mean
from copy import deepcopy
import numpy as np

POP_SIZE        = 240   # population size
MIN_DEPTH       = 0    # minimal initial random tree depth
MAX_DEPTH       = 5    # maximal initial random tree depth
DEPTH_LIMIT     = 10   # maximal tree depth
GENERATIONS     = 250  # maximal number of generations to run evolution
TOURNAMENT_SIZE = 5    # size of tournament for tournament selection
XO_RATE         = 0.8  # crossover rate 
PROB_MUTATION   = 0.1  # per-node mutation probability 
def add(x, y): return x+y
def sub(x, y): return x-y
def mul(x, y): return x*y
def div(x,y): return x/y if y else 0# Consider what issues might arrise with this function
def cos(x): return np.cos(x)
def sin(x): return np.sin(x)
def exp(x) : return np.exp(x)
def absolute(x): return np.abs(x)
def power1(x): return x
def power2(x): return x**2
def power3(x): return x**3
def power4(x): return x**4 
def power_1(x): return 1/x
def power_2(x): return np.sqrt(x) if x > 0 else 1
def power_3(x): return x**(-3) if x > 0 else 1
def power_4(x): return x**(-4) if x > 0 else 1

def f1(x,y):
    return x**2 + y**2 
#Define terminal and non-terminal sets
OPERATIONS = [add, sub, mul, div]
FUNCTIONS1 = [sin,cos]
FUNCTIONS2 = [exp,absolute]
FUNCTIONS3 = [power1,power2,power3,power4,power_1,power_2,power_3,power_4]
FUNCTIONS = FUNCTIONS1 + FUNCTIONS2 + FUNCTIONS3

VARIBLES = ['x','y']
CONSTANTS = [1,2,3,4,10,100, np.pi]

NONTERMINALS = OPERATIONS+FUNCTIONS
TERMINALS = CONSTANTS+VARIBLES
len_t = len(TERMINALS)
len_nt = len(NONTERMINALS)

def generate_dataset(): # generate 101 data points from target_func
    dataset = []
    for x in np.linspace(-1,1,25):
        for y in np.linspace(-1,1,25):
            #dataset.append([x,y, target_func([x,y])])
            dataset.append([x,y, f1([x,y])])
    return dataset


def read_dataset(): # read data from dataset
    f = open('datafile.txt')
    data = f.readlines()
    dataset = []
    for line in data[:-1]:
        x, y, fx = line.split(' ')
        dataset.append([float(x),float(y), float(fx[:-1])])
    x, y, fx = data[-1].split(' ')
    dataset.append([float(x),float(y), float(fx)])
    return dataset

class GPTree:
    def __init__(self, data = None, left = None, right = None,depth = 0, 
                 trig_ = False, power_ = False, exp_ = False, abs_ = False):
        self.data  = data
        self.left  = left
        self.right = right
        self.depth = depth
        self.trig_ = trig_
        #self.func_prob = 
        #self.power_ = power_
        #self.exp_ = exp_
        #self.abs_ = abs_
        
        
    def node_label(self): # string label
        if (self.data in NONTERMINALS):
            return self.data.__name__
        else: 
            return str(self.data)
    
    def print_tree(self, prefix = ""): # textual printout
        print("%s%s" % (prefix, self.node_label()))        
        if self.left:  self.left.print_tree (prefix + "   ")
        if self.right: self.right.print_tree(prefix + "   ")

    def compute_tree(self, x,y): 
        if (self.data in NONTERMINALS):
            if self.data in FUNCTIONS:
                return self.data(self.left.compute_tree(x,y))
            else:
                return self.data(self.left.compute_tree(x,y), self.right.compute_tree(x,y))
        elif self.data == 'x': 
            return x
        elif self.data == 'y': 
            return y
        else: 
            return self.data
        
    def random_tree(self, grow, max_depth = MAX_DEPTH):
        #print(self.depth)
        if self.depth < MIN_DEPTH or (self.depth < max_depth and not grow):
            if self.trig_:
                    self.data = (FUNCTIONS2 + FUNCTIONS3)[randint(0, len(NONTERMINALS)-3)]
            self.data = NONTERMINALS[randint(0, len(NONTERMINALS)-1)]
        
        elif self.depth >= max_depth:
            self.data = TERMINALS[randint(0, len(TERMINALS)-1)]
        else:
            if random() < 0.5:
                self.data = TERMINALS[randint(0,len(TERMINALS)-1)]
            else:
                if self.trig_:
                    self.data = (FUNCTIONS2 + FUNCTIONS3)[randint(0, len(NONTERMINALS)-3)]
                self.data = NONTERMINALS[randint(0, len(NONTERMINALS)-1)]
                
        
        if self.data in NONTERMINALS:
            self.left = GPTree(depth=self.depth+1, trig_ = self.trig_)
            self.left.random_tree(grow = False)
            if self.data in FUNCTIONS:
                self.right = None
            else:
                self.right = GPTree(depth=self.depth+1, trig_ = self.trig_)
                self.right.random_tree(grow = False)
            
            
    def random_tree1(self, grow, max_depth, depth = 0): # create random tree using either grow or full method
        if depth < MIN_DEPTH or (depth < max_depth and not grow): 
            self.data = NONTERMINALS[randint(0, len(NONTERMINALS)-1)]
        elif depth >= max_depth:   
            self.data = TERMINALS[randint(0, len(TERMINALS)-1)]
        else: # intermediate depth, grow
            if random () > 0.5: 
                self.data = TERMINALS[randint(0, len(TERMINALS)-1)]
            else:
                self.data = NONTERMINALS[randint(0, len(FUNCTIONS)-1)]
        if self.data in NONTERMINALS:
            self.left = GPTree()          
            self.left.random_tree(grow, max_depth, depth = depth + 1)
            if not self.data in OPERATIONS:
                self.right = None
            else:
                self.right = GPTree()
                self.right.random_tree(grow, max_depth, depth = depth + 1)
        

    def mutation1(self):
        if random() < PROB_MUTATION: # mutate at this node
            self.random_tree(grow = True, max_depth = 2)
        elif self.left: self.left.mutation()
        elif self.right: self.right.mutation()
        
    def node_mutate(self):
        #  0-3  OPERATIONS
        #  4-15 FUNCTIONS
        # 16-17 VARIBLES
        # 18-24 CONSTANT
        rnd = randint(0, 6)
        """
                if self.data in TERMINALS:
            if self.depth == DEPTH_LIMIT:
                self.data = TERMINALS[randint(0,len_t-1)]
            else:
                
                self.right = GPTree()
                self.right.random_tree(True, max_depth=2)
        """

        if rnd < 2:
            if self.data in TERMINALS:
                if self.depth == DEPTH_LIMIT:
                    self.data = TERMINALS[randint(0,len_t-1)]
                else:
                    self.left = GPTree()
                    self.left.data = TERMINALS[randint(0,len_t-1)]
                    self.right = GPTree()
                    self.right.data = TERMINALS[randint(0,len_t-1)]
            elif not self.right:
                self.right = GPTree()
                self.right.data = TERMINALS[randint(0,len_t-1)]
            self.data = NONTERMINALS[rnd]
        elif rnd < 4:
            self.right = None
            self.data = FUNCTIONS[randint(0,len(FUNCTIONS))]
            if not self.left:
                self.left = GPTree()
                self.left.data = TERMINALS[randint(0,len_t-1)]
        else:
            self.data = TERMINALS[randint(0,len(TERMINALS)-1)]
            self.left = None
            self.right = None
            
        
    def mutation(self):
        if random() < PROB_MUTATION:
            self.node_mutate()
        elif self.left: self.left.mutation()
        elif self.right: self.right.mutation()

    def size(self): # tree size in nodes
        if self.data in TERMINALS: return 1
        l = self.left.size()  if self.left  else 0
        r = self.right.size() if self.right else 0
        return 1 + l + r

    def build_subtree(self): # count is list in order to pass "by reference"
        t = GPTree()
        t.data = self.data
        if self.left:  t.left  = self.left.build_subtree()
        if self.right: t.right = self.right.build_subtree()
        return t
                        
    def scan_tree(self, count, second): # note: count is list, so it's passed "by reference"
        count[0] -= 1            
        if count[0] <= 1: 
            if not second: # return subtree rooted here
                return self.build_subtree()
            else: # glue subtree here
                self.data  = second.data
                self.left  = second.left
                self.right = second.right
        else:  
            ret = None              
            if self.left  and count[0] > 1: ret = self.left.scan_tree(count, second)  
            if self.right and count[0] > 1: ret = self.right.scan_tree(count, second)  
            return ret

    def crossover(self, other): # xo 2 trees at random nodes
        if random() < XO_RATE:
            second = other.scan_tree([randint(1, other.size())], None) # 2nd random subtree
            self.scan_tree([randint(1, self.size()-second.size())], second) # 2nd subtree "glued" inside 1st tree


def fitness(individual, dataset): # inverse mean absolute error over dataset normalized to [0,1]
    return 1 / (1 + mean([abs(np.sum(individual.compute_tree(ds[0],ds[1])) - ds[-1]) for ds in dataset]))

def init_population():
    pop = []
    for md in range(MIN_DEPTH, MAX_DEPTH):
        for i in range(int(POP_SIZE/10)):
            t = GPTree()
            t.random_tree(grow = True, max_depth = md) # grow
            pop.append(t) 
        for i in range(int(POP_SIZE/10)):
            t = GPTree()
            t.random_tree(grow = False, max_depth = md) # full
            pop.append(t) 
    return pop

dataset = read_dataset()
a = GPTree()
a.random_tree(False)

a.print_tree()
a.mutation()
print('---------------------------------')
a.print_tree()
#print(fitness(a,dataset))
#population = init_population()
#population[0].print_tree()
#fitnesses = [fitness(population[i], dataset) for i in range(POP_SIZE)]
#print(fitnesses)

"""

fitnesses = [fitness(population[i], dataset) for i in range(POP_SIZE)]

best_of_run_f = max(fitnesses)
best_of_run = deepcopy(population[fitnesses.index(best_of_run_f)])
best_of_run_gen = 0

best_of_gen = deepcopy(best_of_run)
best_of_gen_f = best_of_run_f
"""