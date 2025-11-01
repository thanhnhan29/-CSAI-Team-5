"""
Test Firefly Algorithm and Cuckoo Search on Knapsack Problem.

This script tests both algorithms on the discrete 0/1 Knapsack optimization problem.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algorithms.firefly import FireflyAlgorithm
from algorithms.cuckoo_search import CuckooSearch
from problems.knapsack import KnapsackProblem
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
    print(f"Testing {algo_name} on Knapsack Problem")
    print(f"{'='*60}")
    
    # Get problem parameters
    bounds = problem.get_bounds()
    n_dim = problem.n_items
    
    # Run optimization
    best_solution, best_fitness, history = algorithm.optimize(
        objective_func=problem.evaluate,
        n_dim=n_dim,
        bounds=bounds
    )
    
    # Get solution details
    solution_info = problem.get_solution_info(best_solution)
    
    # Display results
    print(f"\nResults:")
    print(f"  Best Fitness (negative value): {best_fitness:.2f}")
    print(f"  Total Value: {solution_info['total_value']:.2f}")
    print(f"  Total Weight: {solution_info['total_weight']:.2f} / {solution_info['capacity']:.2f}")
    print(f"  Weight Utilization: {solution_info['weight_utilization']*100:.2f}%")
    print(f"  Items Selected: {solution_info['n_selected']} / {problem.n_items}")
    print(f"  Feasible: {solution_info['is_feasible']}")
    print(f"  Selected Item Indices: {solution_info['selected_items'][:10]}{'...' if len(solution_info['selected_items']) > 10 else ''}")
    
    return best_solution, best_fitness, history


def run_multiple_tests(n_runs=30, n_items=50, max_iter=500, seed=42):
    """
    Run multiple tests to gather statistics.
    
    Parameters:
    -----------
    n_runs : int
        Number of independent runs
    n_items : int
        Number of items in knapsack
    max_iter : int
        Maximum iterations
    seed : int
        Seed for problem generation
        
    Returns:
    --------
    dict
        Results dictionary
    """
    print(f"\n{'#'*60}")
    print(f"Running {n_runs} independent tests")
    print(f"Problem: Knapsack Problem ({n_items} items)")
    print(f"Max Iterations: {max_iter}")
    print(f"{'#'*60}\n")
    
    # Initialize problem (same instance for all runs for fair comparison)
    problem = KnapsackProblem(n_items=n_items, seed=seed)
    
    print(f"Problem Instance:")
    print(f"  Number of Items: {problem.n_items}")
    print(f"  Capacity: {problem.capacity:.2f}")
    print(f"  Total Weight: {np.sum(problem.weights):.2f}")
    print(f"  Total Value: {np.sum(problem.values):.2f}")
    
    # Store results
    results = {
        'FA': {'fitness': [], 'histories': [], 'values': []},
        'CS': {'fitness': [], 'histories': [], 'values': []}
    }
    
    for run in range(n_runs):
        print(f"\n--- Run {run + 1}/{n_runs} ---")
        
        # Firefly Algorithm (Binary mode)
        fa = FireflyAlgorithm(
            n_fireflies=30,
            max_iter=max_iter,
            alpha=0.5,
            beta0=1.0,
            gamma=1.0,
            mode='binary',
            seed=run
        )
        
        fa_solution, fa_fitness, fa_history = run_single_test(fa, problem, "Firefly Algorithm")
        fa_info = problem.get_solution_info(fa_solution)
        results['FA']['fitness'].append(fa_fitness)
        results['FA']['histories'].append(fa_history)
        results['FA']['values'].append(fa_info['total_value'])
        
        # Cuckoo Search (Binary mode)
        cs = CuckooSearch(
            n_nests=30,
            max_iter=max_iter,
            pa=0.25,
            levy_lambda=1.5,
            mode='binary',
            seed=run
        )
        
        cs_solution, cs_fitness, cs_history = run_single_test(cs, problem, "Cuckoo Search")
        cs_info = problem.get_solution_info(cs_solution)
        results['CS']['fitness'].append(cs_fitness)
        results['CS']['histories'].append(cs_history)
        results['CS']['values'].append(cs_info['total_value'])
    
    return results, problem


def print_statistics(results):
    """Print statistical summary of results."""
    print(f"\n{'='*60}")
    print("STATISTICAL SUMMARY (Total Values)")
    print(f"{'='*60}\n")
    
    for algo_name, data in results.items():
        values = np.array(data['values'])
        
        print(f"{algo_name}:")
        print(f"  Best:    {np.max(values):.2f}")  # Max value (best for knapsack)
        print(f"  Worst:   {np.min(values):.2f}")
        print(f"  Mean:    {np.mean(values):.2f}")
        print(f"  Median:  {np.median(values):.2f}")
        print(f"  Std:     {np.std(values):.2f}")
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
    fa_values = results['FA']['values']
    cs_values = results['CS']['values']
    fa_histories = results['FA']['histories']
    cs_histories = results['CS']['histories']
    
    # 1. Convergence curve (best run from each algorithm - by value)
    best_fa_idx = np.argmax(fa_values)  # Max value is best
    best_cs_idx = np.argmax(cs_values)
    
    plot_convergence(
        histories=[fa_histories[best_fa_idx], cs_histories[best_cs_idx]],
        labels=['Firefly Algorithm', 'Cuckoo Search'],
        title='Convergence Curve - Knapsack Problem (Best Run)',
        save_path=f'{output_dir}/knapsack_convergence_best.png',
        show=False
    )
    print("✓ Saved: knapsack_convergence_best.png")
    
    # 2. Multiple runs with confidence intervals
    plot_multiple_runs(
        all_histories={'FA': fa_histories, 'CS': cs_histories},
        algorithm_names=['FA', 'CS'],
        title='Convergence Curves - Knapsack Problem (All Runs)',
        save_path=f'{output_dir}/knapsack_convergence_all.png',
        show=False
    )
    print("✓ Saved: knapsack_convergence_all.png")
    
    # 3. Box plot comparison (using values, not fitness)
    plot_comparison(
        results={'FA': fa_values, 'CS': cs_values},
        metric='distribution',
        title='Algorithm Comparison - Knapsack Problem (Total Values)',
        save_path=f'{output_dir}/knapsack_comparison_box.png',
        show=False
    )
    print("✓ Saved: knapsack_comparison_box.png")
    
    # 4. Bar chart comparison (using values)
    # Convert to negative for visualization (so higher is better visually)
    plot_comparison(
        results={'FA': [-v for v in fa_values], 'CS': [-v for v in cs_values]},
        metric='best',
        title='Best vs Mean Total Value - Knapsack Problem',
        save_path=f'{output_dir}/knapsack_comparison_bar.png',
        show=False
    )
    print("✓ Saved: knapsack_comparison_bar.png")
    
    # 5. Statistics table (using values)
    plot_statistics_table(
        results_dict={'FA': fa_values, 'CS': cs_values},
        save_path=f'{output_dir}/knapsack_statistics.png',
        show=False
    )
    print("✓ Saved: knapsack_statistics.png")
    
    print(f"\nAll visualizations saved to '{output_dir}/' directory")


def main():
    """Main function to run all tests."""
    # Configuration
    N_RUNS = 30
    N_ITEMS = 50
    MAX_ITER = 500
    SEED = 42
    
    # Run tests
    results, problem = run_multiple_tests(
        n_runs=N_RUNS,
        n_items=N_ITEMS,
        max_iter=MAX_ITER,
        seed=SEED
    )
    
    # Print statistics
    print_statistics(results)
    
    # Generate visualizations
    save_visualizations(results)
    
    print(f"\n{'='*60}")
    print("Testing Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
