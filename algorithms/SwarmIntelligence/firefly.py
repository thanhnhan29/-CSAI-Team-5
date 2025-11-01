"""
Firefly Algorithm (FA)

Nature-inspired optimization algorithm based on the flashing behavior of fireflies.
Implements both continuous and discrete (binary) variants.
"""

import numpy as np


class FireflyAlgorithm:
    """
    Firefly Algorithm for optimization.
    
    Can handle both continuous and discrete (binary) optimization problems.
    """
    
    def __init__(self, n_fireflies=20, max_iter=100, alpha=0.5, beta0=1.0, 
                 gamma=1.0, mode='continuous', seed=None):
        """
        Initialize Firefly Algorithm.
        
        Parameters:
        -----------
        n_fireflies : int
            Number of fireflies (population size)
        max_iter : int
            Maximum number of iterations
        alpha : float
            Randomization parameter (step size), typically [0, 1]
        beta0 : float
            Attractiveness at r=0, typically [0, 2]
        gamma : float
            Light absorption coefficient, typically [0.01, 100]
        mode : str
            'continuous' for continuous optimization, 'binary' for discrete
        seed : int, optional
            Random seed for reproducibility
        """
        self.n_fireflies = n_fireflies
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.mode = mode
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # History tracking
        self.best_fitness_history = []
        self.mean_fitness_history = []
    
    def _sigmoid(self, x):
        """Sigmoid transfer function for binary mode."""
        return 1.0 / (1.0 + np.exp(-x))
    
    def _initialize_population(self, n_dim, bounds):
        """
        Initialize firefly population.
        
        Parameters:
        -----------
        n_dim : int
            Problem dimension
        bounds : tuple
            (lower_bounds, upper_bounds)
            
        Returns:
        --------
        numpy.ndarray
            Initial population of shape (n_fireflies, n_dim)
        """
        lower, upper = bounds
        
        if self.mode == 'binary':
            # For binary problems, initialize with probabilities then convert
            population = np.random.rand(self.n_fireflies, n_dim)
        else:
            # For continuous problems
            population = lower + (upper - lower) * np.random.rand(self.n_fireflies, n_dim)
        
        return population
    
    def _calculate_distance(self, firefly_i, firefly_j):
        """
        Calculate Euclidean distance between two fireflies.
        
        Parameters:
        -----------
        firefly_i, firefly_j : numpy.ndarray
            Position vectors
            
        Returns:
        --------
        float
            Euclidean distance
        """
        return np.sqrt(np.sum((firefly_i - firefly_j)**2))
    
    def _attractiveness(self, distance):
        """
        Calculate attractiveness based on distance.
        
        beta(r) = beta0 * exp(-gamma * r^2)
        
        Parameters:
        -----------
        distance : float
            Distance between fireflies
            
        Returns:
        --------
        float
            Attractiveness value
        """
        return self.beta0 * np.exp(-self.gamma * distance**2)
    
    def _move_firefly_continuous(self, firefly_i, firefly_j, beta, bounds):
        """
        Move firefly i towards brighter firefly j (continuous mode).
        
        Parameters:
        -----------
        firefly_i : numpy.ndarray
            Current position of firefly i
        firefly_j : numpy.ndarray
            Position of brighter firefly j
        beta : float
            Attractiveness value
        bounds : tuple
            (lower_bounds, upper_bounds)
            
        Returns:
        --------
        numpy.ndarray
            New position of firefly i
        """
        lower, upper = bounds
        n_dim = len(firefly_i)
        
        # Attraction term
        attraction = beta * (firefly_j - firefly_i)
        
        # Random term
        epsilon = np.random.rand(n_dim) - 0.5
        randomization = self.alpha * epsilon
        
        # New position
        new_position = firefly_i + attraction + randomization
        
        # Clip to bounds
        new_position = np.clip(new_position, lower, upper)
        
        return new_position
    
    def _move_firefly_binary(self, firefly_i, firefly_j, beta, bounds):
        """
        Move firefly i towards brighter firefly j (binary mode).
        
        Parameters:
        -----------
        firefly_i : numpy.ndarray
            Current position of firefly i
        firefly_j : numpy.ndarray
            Position of brighter firefly j
        beta : float
            Attractiveness value
        bounds : tuple
            (lower_bounds, upper_bounds)
            
        Returns:
        --------
        numpy.ndarray
            New position of firefly i
        """
        n_dim = len(firefly_i)
        
        # Attraction in continuous space
        attraction = beta * (firefly_j - firefly_i)
        
        # Random term
        epsilon = np.random.rand(n_dim) - 0.5
        randomization = self.alpha * epsilon
        
        # Update in continuous space
        velocity = attraction + randomization
        continuous_pos = firefly_i + velocity
        
        # Apply sigmoid transfer function
        probabilities = self._sigmoid(continuous_pos)
        
        # Convert to binary
        new_position = (probabilities > np.random.rand(n_dim)).astype(float)
        
        return new_position
    
    def optimize(self, objective_func, n_dim, bounds):
        """
        Run the Firefly Algorithm optimization.
        
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
        fitness = np.array([objective_func(ind) for ind in population])
        
        # Track best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        # History
        self.best_fitness_history = [best_fitness]
        self.mean_fitness_history = [np.mean(fitness)]
        
        # Main loop
        for iteration in range(self.max_iter):
            # For each firefly
            for i in range(self.n_fireflies):
                # Compare with all other fireflies
                for j in range(self.n_fireflies):
                    # If firefly j is brighter (better fitness)
                    if fitness[j] < fitness[i]:
                        # Calculate distance
                        distance = self._calculate_distance(population[i], population[j])
                        
                        # Calculate attractiveness
                        beta = self._attractiveness(distance)
                        
                        # Move firefly i towards j
                        if self.mode == 'binary':
                            population[i] = self._move_firefly_binary(
                                population[i], population[j], beta, bounds
                            )
                        else:
                            population[i] = self._move_firefly_continuous(
                                population[i], population[j], beta, bounds
                            )
                        
                        # Evaluate new position
                        fitness[i] = objective_func(population[i])
            
            # Update best solution
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best_solution = population[current_best_idx].copy()
            
            # Track history
            self.best_fitness_history.append(best_fitness)
            self.mean_fitness_history.append(np.mean(fitness))
            
            # Optional: Reduce randomization over time
            self.alpha *= 0.97
        
        history = {
            'best_fitness': self.best_fitness_history,
            'mean_fitness': self.mean_fitness_history
        }
        
        return best_solution, best_fitness, history
