"""
Test Firefly Algorithm and Cuckoo Search on Rosenbrock Function.

This script tests both algorithms on the continuous Rosenbrock optimization problem.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algorithms.firefly import FireflyAlgorithm
from algorithms.cuckoo_search import CuckooSearch
from problems.rosenbrock import RosenbrockFunction
from utils.visualization import plot_convergence, plot_comparison, plot_multiple_runs, plot_statistics_table


def run_single_test(algorithm, problem, algo_name):
    """
    Run a single test of an algorithm on a problem.
    
    Parameters:
    -----------
    algorithm : object
        Algorithm instance
    problem : object
        Problem instance
    algo_name : str
        Algorithm name for display
        
    Returns:
    --------
    tuple
        (best_solution, best_fitness, history)
    """
    print(f"\n{'='*60}")
    print(f"Testing {algo_name} on Rosenbrock Function")
    print(f"{'='*60}")
    
    # Get problem parameters
    bounds = problem.get_bounds()
    n_dim = problem.n_dim
    
    # Run optimization
    best_solution, best_fitness, history = algorithm.optimize(
        objective_func=problem.evaluate,
        n_dim=n_dim,
        bounds=bounds
    )
    
    # Display results
    print(f"\nResults:")
    print(f"  Best Fitness: {best_fitness:.8f}")
    print(f"  Best Solution: {best_solution}")
    print(f"  Distance from Global Optimum: {np.linalg.norm(best_solution - problem.global_optimum):.8f}")
    print(f"  Final Iteration Best: {history['best_fitness'][-1]:.8f}")
    
    return best_solution, best_fitness, history


def run_multiple_tests(n_runs=30, n_dim=10, max_iter=500):
    """
    Run multiple tests to gather statistics.
    
    Parameters:
    -----------
    n_runs : int
        Number of independent runs
    n_dim : int
        Problem dimension
    max_iter : int
        Maximum iterations
        
    Returns:
    --------
    dict
        Results dictionary
    """
    print(f"\n{'#'*60}")
    print(f"Running {n_runs} independent tests")
    print(f"Problem: Rosenbrock Function ({n_dim}D)")
    print(f"Max Iterations: {max_iter}")
    print(f"{'#'*60}\n")
    
    # Initialize problem
    problem = RosenbrockFunction(n_dim=n_dim)
    
    # Store results
    results = {
        'FA': {'fitness': [], 'histories': []},
        'CS': {'fitness': [], 'histories': []}
    }
    
    for run in range(n_runs):
        print(f"\n--- Run {run + 1}/{n_runs} ---")
        
        # Firefly Algorithm
        fa = FireflyAlgorithm(
            n_fireflies=30,
            max_iter=max_iter,
            alpha=0.5,
            beta0=1.0,
            gamma=1.0,
            mode='continuous',
            seed=run
        )
        
        _, fa_fitness, fa_history = run_single_test(fa, problem, "Firefly Algorithm")
        results['FA']['fitness'].append(fa_fitness)
        results['FA']['histories'].append(fa_history)
        
        # Cuckoo Search
        cs = CuckooSearch(
            n_nests=30,
            max_iter=max_iter,
            pa=0.25,
            levy_lambda=1.5,
            mode='continuous',
            seed=run
        )
        
        _, cs_fitness, cs_history = run_single_test(cs, problem, "Cuckoo Search")
        results['CS']['fitness'].append(cs_fitness)
        results['CS']['histories'].append(cs_history)
    
    return results


def print_statistics(results):
    """Print statistical summary of results."""
    print(f"\n{'='*60}")
    print("STATISTICAL SUMMARY")
    print(f"{'='*60}\n")
    
    for algo_name, data in results.items():
        fitness_values = np.array(data['fitness'])
        
        print(f"{algo_name}:")
        print(f"  Best:    {np.min(fitness_values):.8f}")
        print(f"  Worst:   {np.max(fitness_values):.8f}")
        print(f"  Mean:    {np.mean(fitness_values):.8f}")
        print(f"  Median:  {np.median(fitness_values):.8f}")
        print(f"  Std:     {np.std(fitness_values):.8f}")
        print()


def save_visualizations(results, output_dir='results'):
    """Generate and save all visualizations."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Generating Visualizations...")
    print(f"{'='*60}\n")
    
    # Extract data
    fa_fitness = results['FA']['fitness']
    cs_fitness = results['CS']['fitness']
    fa_histories = results['FA']['histories']
    cs_histories = results['CS']['histories']
    
    # 1. Convergence curve (best run from each algorithm)
    best_fa_idx = np.argmin(fa_fitness)
    best_cs_idx = np.argmin(cs_fitness)
    
    plot_convergence(
        histories=[fa_histories[best_fa_idx], cs_histories[best_cs_idx]],
        labels=['Firefly Algorithm', 'Cuckoo Search'],
        title='Convergence Curve - Rosenbrock Function (Best Run)',
        save_path=f'{output_dir}/rosenbrock_convergence_best.png',
        show=False
    )
    print("✓ Saved: rosenbrock_convergence_best.png")
    
    # 2. Multiple runs with confidence intervals
    plot_multiple_runs(
        all_histories={'FA': fa_histories, 'CS': cs_histories},
        algorithm_names=['FA', 'CS'],
        title='Convergence Curves - Rosenbrock Function (All Runs)',
        save_path=f'{output_dir}/rosenbrock_convergence_all.png',
        show=False
    )
    print("✓ Saved: rosenbrock_convergence_all.png")
    
    # 3. Box plot comparison
    plot_comparison(
        results={'FA': fa_fitness, 'CS': cs_fitness},
        metric='distribution',
        title='Algorithm Comparison - Rosenbrock Function',
        save_path=f'{output_dir}/rosenbrock_comparison_box.png',
        show=False
    )
    print("✓ Saved: rosenbrock_comparison_box.png")
    
    # 4. Bar chart comparison
    plot_comparison(
        results={'FA': fa_fitness, 'CS': cs_fitness},
        metric='best',
        title='Best vs Mean Fitness - Rosenbrock Function',
        save_path=f'{output_dir}/rosenbrock_comparison_bar.png',
        show=False
    )
    print("✓ Saved: rosenbrock_comparison_bar.png")
    
    # 5. Statistics table
    plot_statistics_table(
        results_dict={'FA': fa_fitness, 'CS': cs_fitness},
        save_path=f'{output_dir}/rosenbrock_statistics.png',
        show=False
    )
    print("✓ Saved: rosenbrock_statistics.png")
    
    print(f"\nAll visualizations saved to '{output_dir}/' directory")


def main():
    """Main function to run all tests."""
    # Configuration
    N_RUNS = 30
    N_DIM = 10
    MAX_ITER = 500
    
    # Run tests
    results = run_multiple_tests(n_runs=N_RUNS, n_dim=N_DIM, max_iter=MAX_ITER)
    
    # Print statistics
    print_statistics(results)
    
    # Generate visualizations
    save_visualizations(results)
    
    print(f"\n{'='*60}")
    print("Testing Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
