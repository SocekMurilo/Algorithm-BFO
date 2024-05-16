import numpy as np
from scipy.optimize import rosen


class BFO:
    def __init__(self, n_bacteria, dimension, lower_bound, upper_bound, chemotaxis_steps, swim_length):
        self.n_bacteria = n_bacteria
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.chemotaxis_steps = chemotaxis_steps
        self.swim_length = swim_length
        self.population = np.random.uniform(low=self.lower_bound, high=self.upper_bound, size=(self.n_bacteria, self.dimension))
        self.costs = np.zeros(self.n_bacteria)
        self.best_solution = None
        self.best_cost = np.inf

    def evaluate(self, cost_function):
        for i in range(self.n_bacteria):
            self.costs[i] = cost_function(self.population[i])
            if self.costs[i] < self.best_cost:
                self.best_solution = self.population[i]
                self.best_cost = self.costs[i]

    def chemotaxis(self):
        for i in range(self.n_bacteria):
            for j in range(self.chemotaxis_steps):
                direction = np.random.uniform(low=-1, high=1, size=self.dimension)
                direction /= np.linalg.norm(direction)
                step_size = np.random.uniform(low=0, high=self.swim_length)
                new_position = self.population[i] + step_size * direction
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
                new_cost = cost_function(new_position)
                if new_cost < self.costs[i]:
                    self.population[i] = new_position
                    self.costs[i] = new_cost
                    if new_cost < self.best_cost:
                        self.best_solution = new_position
                        self.best_cost = new_cost
                        if self.best_solution < tol:
                            return

def cost_function(x, y, b):
    return rosen(1, 1, 1)  


#Paramentros
n_bacteria = 20
dimension = 10
lower_bound = -100
upper_bound = 100
chemotaxis_steps = 50
swim_length = 0.1
tol = 1e-8

bfo = BFO(n_bacteria, dimension, lower_bound, upper_bound, chemotaxis_steps, swim_length)
bfo.evaluate(cost_function)
bfo.chemotaxis()

print("Best solution:", bfo.best_solution)
print("Best cost:", bfo.best_cost)
