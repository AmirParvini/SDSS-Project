import numpy as np
# import cupy as cp
import itertools
import random
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt



class rlagent():
    def __init__(self, alpha, gamma, epsilon, cRange, mRange, crossover, mutation):
        
        # the learning rate
        self.alpha = alpha

        # the discount rate
        self.gamma = gamma

        # the exploration rate
        self.epsilon = epsilon
                                   
        # self.actionSpace = list(itertools.product(cRange, mRange))
        self.actionSpace = [(c, m) for c in cRange for m in mRange if abs(c + m - 1) == 0]
        # get the rewards table
        self.rewardSpace = np.array([200,   150,   100,   50,    25,
                                     150,   113,   75,    38,    19,
                                     100,   75,    50,    25,    13,
                                     50,    38,    25,    113,   7,
                                     0,     0,    -10,   -20,   -30,
                                    -1000, -1500, -2000, -2500, -3000])
       
        # a dictionary of all possible states, where the state is the key, and the value is the index for the q table and rewards space
        self.stateSpace = { '(VHC, VHD)': 0, '(VHC, HD)':1,  '(VHC, MD)':2,  '(VHC, LD)':3,  '(VHC, VLD)':4,
                            '(HC, VHD)':5,   '(HC, HD)':6,   '(HC, MD)':7,   '(HC, LD)':8,   '(HC, VLD)':9,
                            '(LC, VHD)':10,  '(LC, HD)':11,  '(LC, MD)':12,  '(LC, LD)':13,  '(LC, VLD)':14,
                            '(VLC, VHD)':15, '(VLC, HD)':16, '(VLC, MD)':17, '(VLC, LD)':18, '(VLC, VLD)':19,
                            '(S, VHD)':20,  '(S, HD)':21,  '(S, MD)':22,  '(S, LD)':23,  '(S, VLD)':24,
                            '(I, VHD)':25,  '(I, HD)':26,  '(I, MD)':27,  '(I, LD)':28,  '(I, VLD)':29}

        self.Q = np.zeros([len(self.stateSpace), len(self.actionSpace)])
        
        
        # ----------------------------------------------- initialization  ------------------------------------------------
        # a variable keeping track of how much rewards it has recieved
        self.collected  = 0

        # create an array to keep count how often each action was taken
        self.actionCount = np.zeros(len(self.actionSpace))

        # the previous fitness variable is initilized with a verh high cost
        self.prevFitness = 10**10
        
        # the current fitness delta
        self.fitness = 0

        # the current diversity index
        self.diversity = 1
        self.div_list = []
        
        # the current reward awarded
        self.reward = 0

        # initialize the first state (high cost, and very high diversity)
        self.currState = 0

        # the first actions are given
        self.action = self.actionSpace.index((crossover, mutation))
    
    
    def __max(self, out, arr):
        # hold any ties found
        ties = []

        # set an initial top value
        top = float('-inf')

        # for each element in the array
        for i in range(len(arr)):

            # if the current value is the new highest value
            if arr[i] > top:

                # then reset the tie list
                ties = []

                # set the new top value
                top = arr[i]

                # add the top value to the tie list
                ties.append([i, arr[i]])

            # else if the current value is tied to the highest value
            elif arr[i] == top:

                # then add it to the tie list
                ties.append([i, arr[i]])
                # ties.append([len(self.actionSpace)-29, len(self.actionSpace)-29])
        
        # pick a random index
        choice = np.random.choice(np.arange(len(ties)))

        # return the desired value
        return ties[choice][out]
    
       
    def __d_fitness(self, fitnesses):
        # get the min fitness of the population
        bestFitness = np.amin(fitnesses)
        # obtaint the difference between the current and previous fitness values
        delta = self.prevFitness - bestFitness
        
        # the difference is divided by the previous fitness to obtain a percentage
        deltaFitness = delta / self.prevFitness
        
        # the current fitness is set as the previous fitness for the next iteration
        self.prevFitness = bestFitness

        # return the fitness imrpovement as a percenetage
        return deltaFitness
    
    
    def __diversity(self, population):
        population = np.array(population)
        sortarr = population[np.lexsort(population.T[::-1])]
        mask = np.empty(population.shape[0], dtype=np.bool_)
        mask[0] = True
        mask[1:] = np.any(sortarr[1:] != sortarr[:-1], axis=1)
        diversity = sortarr[mask].shape[0]/population.shape[0]
        return diversity

    def __reward(self):
        # the reward is look up in the table
        self.reward = self.rewardSpace[self.nextState]

        # the rewards is added to the collection
        self.collected += self.reward

    # used for printing output
    def __findState(self):
        for i in self.stateSpace:
            if self.stateSpace[i] == self.currState:
                return i
    
    
    def __state(self, df, di):
        if df < 0:
            fState = 'I'
        elif df == 0:
            fState = 'S'
        elif df < 0.01:
            fState = 'VLC'
        elif df < 0.05:
            fState = 'LC'
        elif df < 0.25:
            fState = 'HC'
        else:
            fState = 'VHC'

        # an if statment to convert numerical values into into categorical bins
        if di <= 0.2:
            dState = 'VLD'
        elif di <= 0.4:
            dState = 'LD'
        elif di <= 0.6:
            dState = 'MD'
        elif di <= 0.8:
            dState = 'HD'
        else:
            dState = 'VHD'
            
        state = '(' + fState + ', ' + dState + ')'
        self.nextState = self.stateSpace[state]
        self.reward = self.rewardSpace[self.nextState]
        
        
    def initAction(self):
        # reset the action count to disregard the first action
        
        # the action count is updated
        self.actionCount[self.action] += 1
        
        # update the results log
        # self.__results(0)

        # give the enviroment its action (the crossover and mutation probability)
        return self.actionSpace[self.action][0], self.actionSpace[self.action][1]

    def decide(self):
        # randomly decide to explore (with probability epsilon)
        if np.random.random() <= self.epsilon:

            # a random action is chosen
            self.action = int(np.random.randint(low=0, high=len(self.actionSpace)))

        # or exploit (with probability 1 - epsilon)
        else:

            # the max action is chosen
            self.action = int(self.__max(0, (self.Q[self.currState])))
        
        # the action count is updated
        self.actionCount[self.action] += 1
        # print("\n agent.actionCount = ",self.actionCount)

        # print and save the results
        # self.__results(count)

        # give the enviroment its action (the crossover and mutation probability)
        return self.actionSpace[self.action][0], self.actionSpace[self.action][1]
        
        
    # the agent observes the enviroment's response to the agent's action
    def observe(self, population, fitnesses):
        # obtain the population and their fitnesses after an action
        
        # determine the delta of the previous fitness and the current best fitness of the population and the diversity 
        self.fitness = self.__d_fitness(fitnesses)
        self.diversity = self.__diversity(population)
        self.div_list.append(self.diversity)
        # get the new state and rewards
        self.__state(self.fitness, self.diversity)
        self.__reward()

    # the Q table is updated along with other variables for the q learning algorithm
    def updateQlearning(self):
        # update the q table using the bellman equation
        self.Q[self.currState, self.action] += self.alpha * (self.reward + self.gamma * self.__max(1, self.Q[self.nextState]) - self.Q[self.currState, self.action] )

        # update the current state
        self.currState = self.nextState
        
    def df_normalplot(self, df_list):
        sns.histplot(df_list, bins=30, kde=True, stat="density", color="blue", alpha=0.6)
        # اضافه کردن منحنی تابع توزیع نرمال
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, np.mean(df_list), np.std(df_list))
        plt.plot(x, p, 'r', linewidth=2, label="Normal Distribution")

        # نمایش نمودار
        plt.legend()
        plt.title(f"Normal Distribution of Data \n mean = {np.mean(df_list)} \n std = , {np.std(df_list)}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.show()
        
    def diversity_plot(self,g, div_list):
        x = range(g)
        plt.plot(x, div_list)
        

        
        
        
# cRange = np.array(range(1, 11))/10
# mRange = np.array(range(1, 11))/10
# alpha = 0.7
# gamma = 0.1
# epsilon = 0.3
# crossover = 0.8
# mutation = 0.2

# agent = rlagent(alpha, gamma, epsilon, cRange, mRange, crossover, mutation)
# print(agent.actionSpace[-60])