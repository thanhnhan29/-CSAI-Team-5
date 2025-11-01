"""
Cuckoo Search (CS)

Nature-inspired optimization algorithm based on the breeding behavior of cuckoos.
Uses Lévy flights for exploration. Implements both continuous and discrete variants.
"""

import numpy as np
import math


class CuckooSearch:
    """
    Cuckoo Search Algorithm for optimization.
    
    Can handle both continuous and discrete (binary) optimization problems.
    """
    
    def __init__(self, n_nests=25, max_iter=100, pa=0.25, levy_lambda=1.5,
                 mode='continuous', seed=None):
        """
        Initialize Cuckoo Search.
        
        Parameters:
        -----------
        n_nests : int
            Number of nests (population size)
        max_iter : int
            Maximum number of iterations
        pa : float
            Probability of discovering alien eggs [0, 1], typically 0.25
        levy_lambda : float
            Lévy exponent, typically 1.5
        mode : str
            'continuous' for continuous optimization, 'binary' for discrete
        seed : int, optional
            Random seed for reproducibility
        """
        self.n_nests = n_nests
        self.max_iter = max_iter
        self.pa = pa
        self.levy_lambda = levy_lambda
        self.mode = mode
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # History tracking
        self.best_fitness_history = []
        self.mean_fitness_history = []
    
    def _sigmoid(self, x):
        """Sigmoid transfer function for binary mode."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def _levy_flight(self, n_dim):
        """
        Generate Lévy flight step using Mantegna's algorithm.
        
        Parameters:
        -----------
        n_dim : int
            Dimension of the step
            
        Returns:
        --------
        numpy.ndarray
            Lévy flight step vector
        """
        beta = self.levy_lambda
        
        # Calculate sigma
        numerator = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
        denominator = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
        sigma = (numerator / denominator) ** (1 / beta)
        
        # Generate random samples
        u = np.random.normal(0, sigma, n_dim)
        v = np.random.normal(0, 1, n_dim)
        
        # Lévy flight
        step = u / (np.abs(v) ** (1 / beta))
        
        return step
    
    def _initialize_population(self, n_dim, bounds):
        """
        Initialize nest population.
        
        Parameters:
        -----------
        n_dim : int
            Problem dimension
        bounds : tuple
            (lower_bounds, upper_bounds)
            
        Returns:
        --------
        numpy.ndarray
            Initial population of shape (n_nests, n_dim)
        """
        lower, upper = bounds
        
        if self.mode == 'binary':
            # For binary problems, initialize with random binary values
            population = np.random.rand(self.n_nests, n_dim)
            population = (population > 0.5).astype(float)
        else:
            # For continuous problems
            population = lower + (upper - lower) * np.random.rand(self.n_nests, n_dim)
        
        return population
    
    def _get_cuckoo_continuous(self, nest, best_nest, bounds):
        """
        Generate new solution via Lévy flights (continuous mode).
        
        Parameters:
        -----------
        nest : numpy.ndarray
            Current nest position
        best_nest : numpy.ndarray
            Best nest position
        bounds : tuple
            (lower_bounds, upper_bounds)
            
        Returns:
        --------
        numpy.ndarray
            New cuckoo solution
        """
        lower, upper = bounds
        n_dim = len(nest)
        
        # Generate Lévy flight
        levy_step = self._levy_flight(n_dim)
        
        # Step size proportional to problem scale
        step_size = 0.01 * levy_step * (nest - best_nest)
        
        # New solution
        new_solution = nest + step_size
        
        # Clip to bounds
        new_solution = np.clip(new_solution, lower, upper)
        
        return new_solution
    
    def _get_cuckoo_binary(self, nest, best_nest, bounds):
        """
        Generate new solution via Lévy flights (binary mode).
        
        Parameters:
        -----------
        nest : numpy.ndarray
            Current nest position
        best_nest : numpy.ndarray
            Best nest position
        bounds : tuple
            (lower_bounds, upper_bounds)
            
        Returns:
        --------
        numpy.ndarray
            New cuckoo solution (binary)
        """
        n_dim = len(nest)
        
        # Generate Lévy flight in continuous space
        levy_step = self._levy_flight(n_dim)
        
        # Convert binary to continuous-like representation
        continuous_nest = 2 * nest - 1  # Map {0,1} to {-1,1}
        continuous_best = 2 * best_nest - 1
        
        # Step in continuous space
        step_size = 0.1 * levy_step * (continuous_nest - continuous_best)
        continuous_new = continuous_nest + step_size
        
        # Apply sigmoid and convert to binary
        probabilities = self._sigmoid(continuous_new)
        new_solution = (probabilities > np.random.rand(n_dim)).astype(float)
        
        return new_solution
    
    def _empty_nests(self, population, fitness, bounds):
        """
        Abandon a fraction of worst nests and build new ones.
        
        Parameters:
        -----------
        population : numpy.ndarray
            Current population
        fitness : numpy.ndarray
            Fitness values
        bounds : tuple
            (lower_bounds, upper_bounds)
            
        Returns:
        --------
        tuple
            (new_population, new_fitness)
        """
        n_nests, n_dim = population.shape
        
        # Number of nests to abandon
        n_abandon = int(self.pa * n_nests)
        
        if n_abandon == 0:
            return population.copy(), fitness.copy()
        
        # Find worst nests
        worst_indices = np.argsort(fitness)[-n_abandon:]
        
        # Create new nests
        new_population = population.copy()
        
        if self.mode == 'binary':
            # Random binary solutions
            for idx in worst_indices:
                new_population[idx] = (np.random.rand(n_dim) > 0.5).astype(float)
        else:
            # Random continuous solutions
            lower, upper = bounds
            for idx in worst_indices:
                new_population[idx] = lower + (upper - lower) * np.random.rand(n_dim)
        
        return new_population, fitness.copy()
    
    def optimize(self, objective_func, n_dim, bounds):
        """
        Run the Cuckoo Search optimization.
        
        Parameters:
        -----------
        objective_func : callable
            Objective function to minimize
        n_dim : int
            Problem dimension
        bounds : tuple
            (lower_bounds, upper_bounds) as numpy arrays
            
        Returns:
        --------
        tuple
            (best_solution, best_fitness, history)
            - best_solution: Best solution found
            - best_fitness: Best fitness value
            - history: Dictionary with convergence history
        """
        # Initialize population
        population = self._initialize_population(n_dim, bounds)
        
        # Evaluate initial population
        fitness = np.array([objective_func(nest) for nest in population])
        
        # Track best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        # History
        self.best_fitness_history = [best_fitness]
        self.mean_fitness_history = [np.mean(fitness)]
        
        # Main loop
        for iteration in range(self.max_iter):
            # Generate new solutions via Lévy flights
            for i in range(self.n_nests):
                # Get a cuckoo randomly by Lévy flights
                if self.mode == 'binary':
                    new_solution = self._get_cuckoo_binary(
                        population[i], best_solution, bounds
                    )
                else:
                    new_solution = self._get_cuckoo_continuous(
                        population[i], best_solution, bounds
                    )
                
                # Evaluate new solution
                new_fitness = objective_func(new_solution)
                
                # Choose a random nest (say j)
                j = np.random.randint(0, self.n_nests)
                
                # If new solution is better, replace
                if new_fitness < fitness[j]:
                    population[j] = new_solution.copy()
                    fitness[j] = new_fitness
                    
                    # Update best solution
                    if new_fitness < best_fitness:
                        best_fitness = new_fitness
                        best_solution = new_solution.copy()
            
            # Abandon a fraction of worst nests
            population, fitness = self._empty_nests(population, fitness, bounds)
            
            # Re-evaluate abandoned nests
            for i in range(self.n_nests):
                if fitness[i] is None or np.isnan(fitness[i]):
                    fitness[i] = objective_func(population[i])
            
            # Update best solution
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best_solution = population[current_best_idx].copy()
            
            # Track history
            self.best_fitness_history.append(best_fitness)
            self.mean_fitness_history.append(np.mean(fitness))
        
        history = {
            'best_fitness': self.best_fitness_history,
            'mean_fitness': self.mean_fitness_history
        }
        
        return best_solution, best_fitness, history
