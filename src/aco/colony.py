import numpy as np
from .ant import Ant

class ACO:
    def __init__(self, num_ants, num_iter, alpha, beta, rho, q, dist_matrix):
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
        self.best_length = np.inf

    def update_pheromone(self, ants):
        self.pheromone *= (1 - self.rho)
        for ant in ants:
            for i in range(self.num_cities):
                a, b = ant.route[i], ant.route[(i + 1) % self.num_cities]
                delta = self.q / ant.length
                self.pheromone[a][b] += delta
                self.pheromone[b][a] += delta

    def run(self):
        for iteration in range(self.num_iter):
            ants = [Ant(self.num_cities, self.dist_matrix, self.pheromone,
                        self.alpha, self.beta) for _ in range(self.num_ants)]

            for ant in ants:
                ant.construct_route()

            self.update_pheromone(ants)
            min_ant = min(ants, key=lambda a: a.length)
            if min_ant.length < self.best_length:
                self.best_length = min_ant.length
                self.best_route = min_ant.route

            print(f"Iteration {iteration+1}: best length = {self.best_length:.4f}")

        return self.best_route, self.best_length