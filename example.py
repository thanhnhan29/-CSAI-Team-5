"""
Simple Example: Quick start guide for using the algorithms.

This script demonstrates basic usage of both algorithms on simple problems.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from algorithms.firefly import FireflyAlgorithm
from algorithms.cuckoo_search import CuckooSearch
from problems.rosenbrock import RosenbrockFunction
from problems.knapsack import KnapsackProblem


def example_1_rosenbrock():
    """Example 1: Optimize Rosenbrock function with Firefly Algorithm."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Firefly Algorithm on Rosenbrock Function (5D)")
    print("="*60)
    
    # Create problem
    problem = RosenbrockFunction(n_dim=5, bounds=(-5, 5))
    
    # Create algorithm
    fa = FireflyAlgorithm(
        n_fireflies=25,
        max_iter=200,
        alpha=0.5,
        beta0=1.0,
        gamma=1.0,
        mode='continuous',
        seed=42
    )
    
    # Optimize
    print("Running optimization...")
    best_solution, best_fitness, history = fa.optimize(
        objective_func=problem.evaluate,
        n_dim=problem.n_dim,
        bounds=problem.get_bounds()
    )
    
    # Display results
    print(f"\nResults:")
    print(f"  Best fitness found: {best_fitness:.6f}")
    print(f"  Global optimum:     {problem.global_minimum:.6f}")
    print(f"  Best solution:      {best_solution}")
    print(f"  Optimal solution:   {problem.global_optimum}")
    print(f"  Distance to optimum: {np.linalg.norm(best_solution - problem.global_optimum):.6f}")


def example_2_knapsack():
    """Example 2: Optimize Knapsack problem with Cuckoo Search."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Cuckoo Search on Knapsack Problem (30 items)")
    print("="*60)
    
    # Create problem
    problem = KnapsackProblem(n_items=30, seed=42)
    
    print(f"\nProblem details:")
    print(f"  Items: {problem.n_items}")
    print(f"  Capacity: {problem.capacity:.2f}")
    print(f"  Total value (all items): {np.sum(problem.values):.2f}")
    print(f"  Total weight (all items): {np.sum(problem.weights):.2f}")
    
    # Create algorithm
    cs = CuckooSearch(
        n_nests=25,
        max_iter=200,
        pa=0.25,
        levy_lambda=1.5,
        mode='binary',
        seed=42
    )
    
    # Optimize
    print("\nRunning optimization...")
    best_solution, best_fitness, history = cs.optimize(
        objective_func=problem.evaluate,
        n_dim=problem.n_items,
        bounds=problem.get_bounds()
    )
    
    # Get solution details
    info = problem.get_solution_info(best_solution)
    
    # Display results
    print(f"\nResults:")
    print(f"  Total value obtained: {info['total_value']:.2f}")
    print(f"  Total weight used:    {info['total_weight']:.2f} / {info['capacity']:.2f}")
    print(f"  Weight utilization:   {info['weight_utilization']*100:.1f}%")
    print(f"  Items selected:       {info['n_selected']} / {problem.n_items}")
    print(f"  Solution feasible:    {info['is_feasible']}")
    print(f"  Selected items:       {info['selected_items']}")


def example_3_custom_function():
    """Example 3: Use algorithms on a custom function."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Both Algorithms on Custom Sphere Function")
    print("="*60)
    
    # Define custom objective function
    def sphere_function(x):
        """Simple sphere function: f(x) = sum(x^2)"""
        return np.sum(np.array(x)**2)
    
    n_dim = 10
    bounds = (np.full(n_dim, -10), np.full(n_dim, 10))
    
    print(f"\nProblem: Sphere function in {n_dim}D")
    print(f"Bounds: [-10, 10]")
    print(f"Global optimum: f(0, 0, ..., 0) = 0")
    
    # Test Firefly Algorithm
    print("\n--- Firefly Algorithm ---")
    fa = FireflyAlgorithm(n_fireflies=20, max_iter=100, mode='continuous', seed=42)
    fa_solution, fa_fitness, fa_history = fa.optimize(sphere_function, n_dim, bounds)
    print(f"Best fitness: {fa_fitness:.8f}")
    print(f"Solution norm: {np.linalg.norm(fa_solution):.8f}")
    
    # Test Cuckoo Search
    print("\n--- Cuckoo Search ---")
    cs = CuckooSearch(n_nests=20, max_iter=100, mode='continuous', seed=42)
    cs_solution, cs_fitness, cs_history = cs.optimize(sphere_function, n_dim, bounds)
    print(f"Best fitness: {cs_fitness:.8f}")
    print(f"Solution norm: {np.linalg.norm(cs_solution):.8f}")


def main():
    """Run all examples."""
    print("\n" + "#"*60)
    print("# SWARM INTELLIGENCE ALGORITHMS - QUICK EXAMPLES")
    print("#"*60)
    
    # Run examples
    example_1_rosenbrock()
    example_2_knapsack()
    example_3_custom_function()
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)
    print("\nNext steps:")
    print("  - Run full tests: python tests/test_rosenbrock.py")
    print("  - Run all experiments: python experiments/run_experiments.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
