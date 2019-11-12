#!/usr/bin/env python
# coding: utf-8

# In[1]:


# tiny genetic programming by © moshe sipper, www.moshesipper.com
import graphviz
from random import random, randint, seed
from statistics import mean
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython.display import Image, display
from graphviz import Digraph, Source 

POP_SIZE        = 500   # population size
MIN_DEPTH       = 2    # minimal initial random tree depth
MAX_DEPTH       = 5    # maximal initial random tree depth
GENERATIONS     = 500  # maximal number of generations to run evolution
TOURNAMENT_SIZE = 40    # size of tournament for tournament selection
XO_RATE         = 0.8  # crossover rate 
PROB_MUTATION   = 0  # per-node mutation probability 
how_many_primes = 40  # number of primes to use for fitness check
creation_max_depth = False # grow or max depth for tree initialisation False = Grow / True = Max Depth

userinput = list(range(0, how_many_primes))

k = 1

def add(x, y): return x + y
def sub(x, y): return x - y
def mul(x, y): return x * y
FUNCTIONS = [add, sub, mul]
TERMINALS = ['x', -10, -5, -2, -1, 0, 1, 2, 5, 10] 

def is_prime_number(x): #checks to see if a given number is prime
    if x >= 2:
        for y in range(2,x):
            if not ( x % y ):
                return False
    else:
        return False
    return True

def generateprimes(userinput): #allows the generation a sequential list of primes that is a specific length
    prime_numbers = 0
    dataset = []
    count = 2
    while prime_numbers < how_many_primes:
        if is_prime_number(count):
            count +=1
            prime_numbers +=1
            dataset.append(count)
        if is_prime_number(count) == False:
            count +=1
        if prime_numbers == how_many_primes:
            print ("congratulations you generated " + str(prime_numbers) + " prime numbers")
    return dataset


# In[ ]:





# In[2]:


generateprimes(userinput)


# def generate_dataset(): # generate 101 data points from euler
#     dataset = []
#     for x in range(0, how_many_numbers): 
#         x /= 40
#         dataset.append([x, userinput(x)])
#     return dataset

# In[3]:


class GPTree:
    def __init__(self, data = None, left = None, right = None):
        self.data  = data
        self.left  = left
        self.right = right
        
    def node_label(self): # string label
        if (self.data in FUNCTIONS):
            return self.data.__name__
        else: 
            return str(self.data)
    
    def print_tree(self, prefix = ""): # textual printout
        print("%s%s" % (prefix, self.node_label()))        
        if self.left:  self.left.print_tree(prefix + "Left   ")
        if self.right: self.right.print_tree(prefix + "Right   ")

    def compute_tree(self, x): 
        if (self.data in FUNCTIONS): 
            return self.data(self.left.compute_tree(x), self.right.compute_tree(x))
        elif self.data == 'x': return x
        else: return self.data
            
    def random_tree(self, grow, max_depth, depth = 0): # create random tree using either grow or full method
        if depth < MIN_DEPTH or (depth < max_depth and not grow): 
            self.data = FUNCTIONS[randint(0, len(FUNCTIONS)-1)]
        elif depth >= max_depth:   
            self.data = TERMINALS[randint(0, len(TERMINALS)-1)]
        else: # intermediate depth, grow
            if random () > 0.5: 
                self.data = TERMINALS[randint(0, len(TERMINALS)-1)]
            else:
                self.data = FUNCTIONS[randint(0, len(FUNCTIONS)-1)]
        if self.data in FUNCTIONS:
            self.left = GPTree()          
            self.left.random_tree(grow, max_depth, depth = depth + 1)            
            self.right = GPTree()
            self.right.random_tree(grow, max_depth, depth = depth + 1)

    def mutation(self):
        if random() < PROB_MUTATION: # mutate at this node
            self.random_tree(grow = True, max_depth = 2)
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
            self.scan_tree([randint(1, self.size())], second) # 2nd subtree "glued" inside 1st tree
                    
    def draw(self, dot, count): # dot & count are lists in order to pass "by reference" 
        node_name = str(count[0])
        dot[0].node(node_name, self.node_label())
        if self.left:
            count[0] += 1
            dot[0].edge(node_name, str(count[0]))
            self.left.draw(dot, count)
        if self.right:
            count[0] += 1
            dot[0].edge(node_name, str(count[0]))
            self.right.draw(dot, count)
            
    def draw_tree(self, fname, footer):
        dot = [Digraph()]
        dot[0].attr(kw='graph', label = footer)
        count = [0]
        self.draw(dot, count)
        Source(dot[0], filename = fname + ".gv", format="png").render()
        display(Image(filename = fname + ".gv.png"))
        
#end class GPTree


# In[ ]:





# Initialise Tree Population

# In[4]:


def init_population(): # ramped half-and-half
    pop = []
    if creation_max_depth == True:
        for md in range(5, MAX_DEPTH + 1):
            for i in range(int(POP_SIZE)):
                t = GPTree()
                t.random_tree(grow = False, max_depth = md) # grow
                pop.append(t) 
    else:
        for md in range(5, MAX_DEPTH + 1):
            for i in range(int(POP_SIZE)):
                t = GPTree()
                t.random_tree(grow = True, max_depth = md) # grow
                pop.append(t) 
        #for i in range(int(POP_SIZE/6)):
         #   t = GPTree()
          #  t.random_tree(grow = False, max_depth = md) # full
           # pop.append(t) 
    return pop


# In[5]:


POP_SIZE


# In[6]:


population = init_population()


# In[7]:


def population_sizes():
    for i in population:
        print(GPTree.size(i))
        
# gives us the size of each of the trees in the population


# In[ ]:





# In[ ]:





# In[8]:


def population_count():
    count = 0
    for i in population:
        count += 1
    print(count)   

# does a simple count of the current population


# In[9]:


population_count()


# In[10]:


for i in population:
    print(GPTree.print_tree(i))


# Plot Population Info

# In[11]:


def prepare_plots():
    fig, axarr = plt.subplots(2, sharex=True)
    fig.canvas.set_window_title('EVOLUTIONARY PROGRESS')
    fig.subplots_adjust(hspace = 0.5)
    axarr[0].set_title('error', fontsize=14)
    axarr[1].set_title('mean size', fontsize=14)
    plt.xlabel('generation', fontsize=18)
    plt.ion() # interactive mode for plot
    axarr[0].set_xlim(0, GENERATIONS)
    axarr[0].set_ylim(0, 1) # fitness range
    xdata = []
    ydata = [ [], [] ]
    line = [None, None]
    line[0], = axarr[0].plot(xdata, ydata[0], 'b-') # 'b-' = blue line    
    line[1], = axarr[1].plot(xdata, ydata[1], 'r-') # 'r-' = red line
    return axarr, line, xdata, ydata

def plot(axarr, line, xdata, ydata, gen, pop, errors, max_mean_size):
    xdata.append(gen)
    ydata[0].append(min(errors))
    line[0].set_xdata(xdata)
    line[0].set_ydata(ydata[0])
    sizes = [ind.size() for ind in pop]
    if mean(sizes) > max_mean_size[0]:
        max_mean_size[0] = mean(sizes)
        axarr[1].set_ylim(0, max_mean_size[0])
    ydata[1].append(mean(sizes))
    line[1].set_xdata(xdata)
    line[1].set_ydata(ydata[1])
    plt.draw()  
    plt.pause(0.01)


# In[12]:


def draw_initial_pop():
    tree = 1
    for individual in population:
        print(GPTree.draw_tree(individual, "Header", "Tree " + str(tree)))
        tree += 1
        
# This function allows us to draw the initial bunch of trees


# In[13]:


#draw_initial_pop()


# # Fitness Function

# In[14]:


k = 1


# In[15]:


test_func = k*k + k + 17


# In[16]:


def fitness_check_sequential(individual):
    k = 1
    individual_fitness = 0
    while k <= how_many_primes:
        if calc_output(k) == calc_output(k + 1):
            return individual_fitness
        if tree_check_prime(k) == False:
            return individual_fitness
        if tree_check_prime(k) == True:
            individual_fitness += 1
            k += 1
    return individual_fitness   


# In[ ]:





# In[ ]:





# In[17]:


def calc_output(k):
    return GPTree.compute_tree(individual, k)


# In[18]:


def tree_check_prime(k):
    return is_prime_number(calc_output(k))


# In[19]:


for individual in population:
    print(calc_output(k))


# In[ ]:





# In[ ]:





# In[20]:


def fitness(individual):
    k = 1
    individual_fitness = 0
    while k <= how_many_primes:
        if calc_output(k) == calc_output(k + 1):
            return individual_fitness
        if tree_check_prime(k) == False:
            return individual_fitness
        if tree_check_prime(k) == True:
            individual_fitness += 1
            k += 1
    return individual_fitness       


# In[21]:


for individual in population:
    if calc_output(k) == calc_output(k + 1):
        print ("no k value")
    else:
        print ("has k")


# In[22]:


for individual in population:
    print(fitness_check_sequential(individual))


# In[23]:


def test_check_sequential(): 
    k=1
    individual_fitness = 0
    while k <= how_many_primes:
        if is_prime_number(test_func) == False:
            return individual_fitness
        if is_prime_number(test_func) == True:
            individual_fitness += 1
            k += 1
    return individual_fitness


# In[24]:


for individual in population:
    print(fitness_check_sequential(individual))


# In[25]:


for individual in population:
    print(tree_check_prime(k))


# In[26]:


for individual in population:
    print("Fitness = " + str(fitness(individual)))


# In[27]:


fitnesses = []
for individual in population:
    fitnesses.append(fitness(individual))


# # Selection Function

# In[28]:


def selection(population, fitnesses): # select one individual using tournament selection
    tournament = [randint(0, len(population)-1) for i in range(TOURNAMENT_SIZE)] # select tournament contenders
    tournament_fitnesses = [fitnesses[tournament[i]] for i in range(TOURNAMENT_SIZE)]
    return deepcopy(population[tournament[tournament_fitnesses.index(max(tournament_fitnesses))]]) 
            


# In[29]:


def draw_initial_pop():
    tree = 1
    for i in population:
        print(GPTree.draw_tree(i, "Header","Tree # " + str(tree)))
        tree +=1
        
# This function allows us to draw the initial bunch of trees


# In[30]:


fitnesses = []
for individual in population:
    fitnesses.append(fitness(individual))


# In[31]:


print(fitnesses)


# In[32]:


fitnesses.index(max(fitnesses))


# In[33]:


max(fitnesses)


# Main Program

# In[34]:


seed() # init internal state of random number generator
generateprimes(userinput)
population = init_population() 
best_of_run = None
best_of_run_f = 0
best_of_run_gen = 0
fitnesses = []
for individual in population:
    fitnesses.append(fitness(individual))
print(fitnesses)

        


# In[35]:


# go evolution!
for gen in range(GENERATIONS):        
    nextgen_population=[]
    for i in range(POP_SIZE):
        parent1 = selection(population, fitnesses)
        parent2 = selection(population, fitnesses)
        parent1.crossover(parent2)
        parent1.mutation()
        nextgen_population.append(parent1)
    population = nextgen_population
    fitnesses = []
    for individual in population:
        fitnesses.append(fitness(individual))
    if max(fitnesses) >= best_of_run_f:
        best_of_run_f = max(fitnesses)
        best_of_run_gen = gen
        best_of_run = deepcopy(population[fitnesses.index(max(fitnesses))])
        print("________________________")
        print("gen:", gen, ", best_of_run_f:", max(fitnesses), ", best_of_run:") 
        #best_of_run.print_tree()
    if best_of_run_f == how_many_primes: break   

print("\n\n_________________________________________________\nEND OF RUN\nbest_of_run attained at gen " + str(best_of_run_gen) +      " and has f=" + str(best_of_run_f))
best_of_run.print_tree()


# In[ ]:





# In[36]:


GPTree.draw_tree(best_of_run, "best of run", "Best of run at generation " + str(best_of_run_gen) + " managed to generate " + str(best_of_run_f) + " primes" )


# In[37]:





# In[ ]:





# In[ ]:




