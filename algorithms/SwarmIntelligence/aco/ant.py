import numpy as np
import random

class Ant:
    def __init__(self, num_cities, dist_matrix, pheromone, compute_reward, stop_condition, alpha, beta):
        self.num_cities = num_cities
        self.dist_matrix = dist_matrix
        self.pheromone = pheromone
        self.alpha = alpha
        self.beta = beta
        self.route = []
        self.reward = 0.0
        self.compute_reward = compute_reward
        self.stop_condition = stop_condition

    def select_next_city(self, current_city, visited):
        probs = np.zeros(self.num_cities)
        for j in range(self.num_cities):
            if j not in visited:
                tau = self.pheromone[current_city][j] ** self.alpha
                eta = (1.0 / (self.dist_matrix[current_city][j] + 1e-10)) ** self.beta
                probs[j] = tau * eta
        if probs.sum() == 0:
            return random.choice(list(set(range(self.num_cities)) - set(visited)))
        probs /= probs.sum()
        return np.random.choice(range(self.num_cities), p=probs)

    def construct_route(self):
        self.route = [random.randint(0, self.num_cities - 1)]
        while self.stop_condition(self.route):
            next_city = self.select_next_city(self.route[-1], self.route)
            self.route.append(next_city)
        self.reward = self.compute_reward(self.route)

    def compute(self):
        return self.compute_reward(self.route)
