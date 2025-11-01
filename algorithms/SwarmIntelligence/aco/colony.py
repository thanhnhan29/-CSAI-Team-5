import numpy as np
from .ant import Ant

class ACO:
    def __init__(self, num_ants, num_iter, alpha, beta, rho, q, dist_matrix, 
                 compute_reward=None, stop_condition=None):
        self.num_ants = num_ants
        self.num_iter = num_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.dist_matrix = dist_matrix
        self.num_cities = len(dist_matrix)
        self.pheromone = np.ones((self.num_cities, self.num_cities))
        self.best_route = None
        self.best_reward = -1 * np.inf
        
        self.compute_reward = compute_reward if compute_reward else self._default_tsp_reward
        self.stop_condition = stop_condition if stop_condition else self._default_tsp_stop_condition
    
    def _default_tsp_reward(self, route):
        """Default reward function for TSP: negative of total distance"""
        distance = 0
        for i in range(len(route)):
            distance += self.dist_matrix[route[i]][route[(i + 1) % len(route)]]
        return -distance
    
    def _default_tsp_stop_condition(self, route):
        """Default stop condition for TSP: all cities visited"""
        return len(route) < self.num_cities

    def update_pheromone(self, ants):
        self.pheromone *= (1 - self.rho)
        for ant in ants:
            for i in range(len(ant.route) - 1):
                a, b = ant.route[i], ant.route[i + 1]
                delta = abs(ant.reward)
                self.pheromone[a][b] += delta
                self.pheromone[b][a] += delta

    def run(self):
        for iteration in range(self.num_iter):
            ants = [Ant(self.num_cities, self.dist_matrix, self.pheromone,
                        self.compute_reward, self.stop_condition, self.alpha, self.beta) for _ in range(self.num_ants)]

            for ant in ants:
                ant.construct_route()

            self.update_pheromone(ants)
            max_ant = max(ants, key=lambda a: a.reward)
            if max_ant.reward > self.best_reward:
                self.best_reward = max_ant.reward
                self.best_route = max_ant.route

            print(f"Iteration {iteration+1}: best reward = {self.best_reward:.4f}")

        return self.best_route, self.best_reward