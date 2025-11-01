"""
Knapsack Problem for Discrete Optimization

The 0/1 Knapsack Problem: maximize total value subject to weight constraint.
"""

import numpy as np


class KnapsackProblem:
    """0/1 Knapsack Problem for discrete optimization."""
    
    def __init__(self, weights=None, values=None, capacity=None, n_items=None, seed=None):
        """
        Initialize Knapsack Problem.
        
        Parameters:
        -----------
        weights : numpy.ndarray, optional
            Weight of each item
        values : numpy.ndarray, optional
            Value of each item
        capacity : float, optional
            Maximum capacity of knapsack
        n_items : int, optional
            Number of items (used if weights/values not provided)
        seed : int, optional
            Random seed for generating instance
        """
        if weights is not None and values is not None and capacity is not None:
            # Use provided instance
            self.weights = np.asarray(weights)
            self.values = np.asarray(values)
            self.capacity = capacity
            self.n_items = len(weights)
        elif n_items is not None:
            # Generate random instance
            self.n_items = n_items
            self._generate_instance(n_items, seed)
        else:
            raise ValueError("Either provide (weights, values, capacity) or n_items")
        
        # Validate
        if len(self.weights) != len(self.values):
            raise ValueError("weights and values must have same length")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
    
    def _generate_instance(self, n_items, seed=None):
        """
        Generate a random knapsack instance.
        
        Parameters:
        -----------
        n_items : int
            Number of items
        seed : int, optional
            Random seed
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random weights and values
        self.weights = np.random.randint(1, 50, size=n_items).astype(float)
        self.values = np.random.randint(1, 100, size=n_items).astype(float)
        
        # Set capacity to approximately 50% of total weight
        self.capacity = 0.5 * np.sum(self.weights)
    
    def evaluate(self, solution):
        """
        Evaluate a solution (binary vector).
        
        Parameters:
        -----------
        solution : numpy.ndarray
            Binary vector indicating which items to include (1) or exclude (0)
            
        Returns:
        --------
        float
            Total value (negative if constraint violated, includes penalty)
        """
        solution = np.asarray(solution)
        
        # Ensure binary solution
        binary_solution = (solution >= 0.5).astype(int)
        
        # Calculate total weight and value
        total_weight = np.sum(binary_solution * self.weights)
        total_value = np.sum(binary_solution * self.values)
        
        # Apply penalty if capacity exceeded
        if total_weight > self.capacity:
            # Heavy penalty for constraint violation
            penalty = (total_weight - self.capacity) * np.max(self.values) * 10
            return -(penalty)  # Return negative to minimize
        
        # Return negative value (for minimization in optimization algorithms)
        return -total_value
    
    def __call__(self, solution):
        """Allow calling the object as a function."""
        return self.evaluate(solution)
    
    def get_bounds(self):
        """
        Get the search space bounds (0 to 1 for binary encoding).
        
        Returns:
        --------
        tuple
            (lower_bounds, upper_bounds) as numpy arrays
        """
        lower = np.zeros(self.n_items)
        upper = np.ones(self.n_items)
        return lower, upper
    
    def decode_solution(self, solution):
        """
        Decode a solution to binary format.
        
        Parameters:
        -----------
        solution : numpy.ndarray
            Continuous or binary solution vector
            
        Returns:
        --------
        numpy.ndarray
            Binary solution (0 or 1)
        """
        return (np.asarray(solution) >= 0.5).astype(int)
    
    def is_feasible(self, solution):
        """
        Check if a solution is feasible (satisfies capacity constraint).
        
        Parameters:
        -----------
        solution : numpy.ndarray
            Solution vector
            
        Returns:
        --------
        bool
            True if solution is feasible
        """
        binary_solution = self.decode_solution(solution)
        total_weight = np.sum(binary_solution * self.weights)
        return total_weight <= self.capacity
    
    def repair_solution(self, solution):
        """
        Repair an infeasible solution using a greedy approach.
        
        Parameters:
        -----------
        solution : numpy.ndarray
            Potentially infeasible solution
            
        Returns:
        --------
        numpy.ndarray
            Repaired feasible solution
        """
        binary_solution = self.decode_solution(solution).astype(float)
        
        # If already feasible, return as is
        if self.is_feasible(binary_solution):
            return binary_solution
        
        # Remove items with lowest value-to-weight ratio until feasible
        value_weight_ratio = self.values / self.weights
        indices = np.argsort(value_weight_ratio)  # Sort by ratio (ascending)
        
        current_weight = np.sum(binary_solution * self.weights)
        
        for idx in indices:
            if current_weight <= self.capacity:
                break
            if binary_solution[idx] == 1:
                binary_solution[idx] = 0
                current_weight -= self.weights[idx]
        
        return binary_solution
    
    def get_solution_info(self, solution):
        """
        Get detailed information about a solution.
        
        Parameters:
        -----------
        solution : numpy.ndarray
            Solution vector
            
        Returns:
        --------
        dict
            Dictionary with solution details
        """
        binary_solution = self.decode_solution(solution)
        total_weight = np.sum(binary_solution * self.weights)
        total_value = np.sum(binary_solution * self.values)
        is_feasible = total_weight <= self.capacity
        
        return {
            'binary_solution': binary_solution,
            'selected_items': np.where(binary_solution == 1)[0],
            'n_selected': np.sum(binary_solution),
            'total_weight': total_weight,
            'total_value': total_value,
            'capacity': self.capacity,
            'weight_utilization': total_weight / self.capacity,
            'is_feasible': is_feasible
        }
