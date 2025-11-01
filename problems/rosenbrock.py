"""
Rosenbrock Function for Continuous Optimization

The Rosenbrock function is a classic test problem for optimization algorithms.
f(x) = sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]

Global minimum: f(1, 1, ..., 1) = 0
"""

import numpy as np


class RosenbrockFunction:
    """Rosenbrock benchmark function for continuous optimization."""
    
    def __init__(self, n_dim=2, bounds=None):
        """
        Initialize Rosenbrock function.
        
        Parameters:
        -----------
        n_dim : int
            Number of dimensions
        bounds : tuple or list
            Search space bounds (min, max). Default is (-5, 5)
        """
        self.n_dim = n_dim
        self.bounds = bounds if bounds is not None else (-5, 5)
        self.global_optimum = np.ones(n_dim)
        self.global_minimum = 0.0
        
    def evaluate(self, x):
        """
        Evaluate the Rosenbrock function at point x.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Input vector of shape (n_dim,)
            
        Returns:
        --------
        float
            Function value at x
        """
        x = np.asarray(x)
        
        # Rosenbrock function: sum of 100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2
        sum_sq_term = 100.0 * (x[1:] - x[:-1]**2)**2
        sq_term = (1.0 - x[:-1])**2
        
        return np.sum(sum_sq_term + sq_term)
    
    def __call__(self, x):
        """Allow calling the object as a function."""
        return self.evaluate(x)
    
    def get_bounds(self):
        """
        Get the search space bounds.
        
        Returns:
        --------
        tuple
            (lower_bounds, upper_bounds) as numpy arrays
        """
        if isinstance(self.bounds, tuple):
            lower = np.full(self.n_dim, self.bounds[0])
            upper = np.full(self.n_dim, self.bounds[1])
        else:
            lower = np.array([b[0] for b in self.bounds])
            upper = np.array([b[1] for b in self.bounds])
        
        return lower, upper
    
    def is_valid(self, x):
        """
        Check if solution x is within bounds.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Solution vector
            
        Returns:
        --------
        bool
            True if x is within bounds
        """
        lower, upper = self.get_bounds()
        return np.all(x >= lower) and np.all(x <= upper)
    
    def clip_to_bounds(self, x):
        """
        Clip solution x to be within bounds.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Solution vector
            
        Returns:
        --------
        numpy.ndarray
            Clipped solution
        """
        lower, upper = self.get_bounds()
        return np.clip(x, lower, upper)
