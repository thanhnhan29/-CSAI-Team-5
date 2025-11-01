"""
Main Experiment Runner

This script runs all experiments: testing Firefly Algorithm and Cuckoo Search
on both Rosenbrock Function and Knapsack Problem.
"""

import numpy as np
import sys
import os
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algorithms.firefly import FireflyAlgorithm
from algorithms.cuckoo_search import CuckooSearch
from problems.rosenbrock import RosenbrockFunction
from problems.knapsack import KnapsackProblem
from utils.visualization import (plot_convergence, plot_comparison, 
                                 plot_multiple_runs, plot_statistics_table,
                                 plot_rosenbrock_surface)


def run_rosenbrock_experiments(n_runs=30, n_dim=10, max_iter=500):
    """
    Run experiments on Rosenbrock Function.
    
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
    print(f"\n{'#'*70}")
    print(f"# EXPERIMENT 1: ROSENBROCK FUNCTION")
    print(f"{'#'*70}")
    print(f"Configuration:")
    print(f"  - Dimension: {n_dim}")
    print(f"  - Independent Runs: {n_runs}")
    print(f"  - Max Iterations: {max_iter}")
    print(f"  - Population Size: 30")
    
    problem = RosenbrockFunction(n_dim=n_dim)
    bounds = problem.get_bounds()
    
    results = {
        'FA': {'fitness': [], 'histories': [], 'times': []},
        'CS': {'fitness': [], 'histories': [], 'times': []}
    }
    
    for run in range(n_runs):
        print(f"\n  Run {run + 1}/{n_runs}...", end=" ")
        
        # Firefly Algorithm
        fa = FireflyAlgorithm(n_fireflies=30, max_iter=max_iter, alpha=0.5,
                             beta0=1.0, gamma=1.0, mode='continuous', seed=run)
        
        start_time = time.time()
        _, fa_fitness, fa_history = fa.optimize(problem.evaluate, n_dim, bounds)
        fa_time = time.time() - start_time
        
        results['FA']['fitness'].append(fa_fitness)
        results['FA']['histories'].append(fa_history)
        results['FA']['times'].append(fa_time)
        
        # Cuckoo Search
        cs = CuckooSearch(n_nests=30, max_iter=max_iter, pa=0.25,
                         levy_lambda=1.5, mode='continuous', seed=run)
        
        start_time = time.time()
        _, cs_fitness, cs_history = cs.optimize(problem.evaluate, n_dim, bounds)
        cs_time = time.time() - start_time
        
        results['CS']['fitness'].append(cs_fitness)
        results['CS']['histories'].append(cs_history)
        results['CS']['times'].append(cs_time)
        
        print(f"FA: {fa_fitness:.6f} ({fa_time:.2f}s), CS: {cs_fitness:.6f} ({cs_time:.2f}s)")
    
    return results


def run_knapsack_experiments(n_runs=30, n_items=50, max_iter=500, seed=42):
    """
    Run experiments on Knapsack Problem.
    
    Parameters:
    -----------
    n_runs : int
        Number of independent runs
    n_items : int
        Number of items
    max_iter : int
        Maximum iterations
    seed : int
        Problem instance seed
        
    Returns:
    --------
    tuple
        (results, problem)
    """
    print(f"\n{'#'*70}")
    print(f"# EXPERIMENT 2: KNAPSACK PROBLEM")
    print(f"{'#'*70}")
    print(f"Configuration:")
    print(f"  - Number of Items: {n_items}")
    print(f"  - Independent Runs: {n_runs}")
    print(f"  - Max Iterations: {max_iter}")
    print(f"  - Population Size: 30")
    
    problem = KnapsackProblem(n_items=n_items, seed=seed)
    bounds = problem.get_bounds()
    
    print(f"\nProblem Instance:")
    print(f"  - Capacity: {problem.capacity:.2f}")
    print(f"  - Total Weight: {np.sum(problem.weights):.2f}")
    print(f"  - Total Value: {np.sum(problem.values):.2f}")
    
    results = {
        'FA': {'fitness': [], 'histories': [], 'values': [], 'times': []},
        'CS': {'fitness': [], 'histories': [], 'values': [], 'times': []}
    }
    
    for run in range(n_runs):
        print(f"\n  Run {run + 1}/{n_runs}...", end=" ")
        
        # Firefly Algorithm (Binary mode)
        fa = FireflyAlgorithm(n_fireflies=30, max_iter=max_iter, alpha=0.5,
                             beta0=1.0, gamma=1.0, mode='binary', seed=run)
        
        start_time = time.time()
        fa_solution, fa_fitness, fa_history = fa.optimize(problem.evaluate, n_items, bounds)
        fa_time = time.time() - start_time
        
        fa_info = problem.get_solution_info(fa_solution)
        results['FA']['fitness'].append(fa_fitness)
        results['FA']['histories'].append(fa_history)
        results['FA']['values'].append(fa_info['total_value'])
        results['FA']['times'].append(fa_time)
        
        # Cuckoo Search (Binary mode)
        cs = CuckooSearch(n_nests=30, max_iter=max_iter, pa=0.25,
                         levy_lambda=1.5, mode='binary', seed=run)
        
        start_time = time.time()
        cs_solution, cs_fitness, cs_history = cs.optimize(problem.evaluate, n_items, bounds)
        cs_time = time.time() - start_time
        
        cs_info = problem.get_solution_info(cs_solution)
        results['CS']['fitness'].append(cs_fitness)
        results['CS']['histories'].append(cs_history)
        results['CS']['values'].append(cs_info['total_value'])
        results['CS']['times'].append(cs_time)
        
        print(f"FA: {fa_info['total_value']:.2f} ({fa_time:.2f}s), CS: {cs_info['total_value']:.2f} ({cs_time:.2f}s)")
    
    return results, problem


def print_summary(rosenbrock_results, knapsack_results):
    """Print comprehensive summary of all experiments."""
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE RESULTS SUMMARY")
    print(f"{'='*70}\n")
    
    # Rosenbrock Results
    print("1. ROSENBROCK FUNCTION")
    print("-" * 70)
    for algo in ['FA', 'CS']:
        fitness = np.array(rosenbrock_results[algo]['fitness'])
        times = np.array(rosenbrock_results[algo]['times'])
        print(f"\n{algo}:")
        print(f"  Best Fitness:      {np.min(fitness):.8f}")
        print(f"  Mean Fitness:      {np.mean(fitness):.8f}")
        print(f"  Std Fitness:       {np.std(fitness):.8f}")
        print(f"  Avg Time per Run:  {np.mean(times):.2f}s")
    
    # Statistical comparison
    fa_fitness = np.array(rosenbrock_results['FA']['fitness'])
    cs_fitness = np.array(rosenbrock_results['CS']['fitness'])
    print(f"\nComparison:")
    print(f"  FA wins: {np.sum(fa_fitness < cs_fitness)} runs")
    print(f"  CS wins: {np.sum(cs_fitness < fa_fitness)} runs")
    print(f"  Ties:    {np.sum(fa_fitness == cs_fitness)} runs")
    
    # Knapsack Results
    print(f"\n{'='*70}")
    print("2. KNAPSACK PROBLEM")
    print("-" * 70)
    for algo in ['FA', 'CS']:
        values = np.array(knapsack_results[algo]['values'])
        times = np.array(knapsack_results[algo]['times'])
        print(f"\n{algo}:")
        print(f"  Best Value:        {np.max(values):.2f}")
        print(f"  Mean Value:        {np.mean(values):.2f}")
        print(f"  Std Value:         {np.std(values):.2f}")
        print(f"  Avg Time per Run:  {np.mean(times):.2f}s")
    
    # Statistical comparison
    fa_values = np.array(knapsack_results['FA']['values'])
    cs_values = np.array(knapsack_results['CS']['values'])
    print(f"\nComparison:")
    print(f"  FA wins: {np.sum(fa_values > cs_values)} runs")
    print(f"  CS wins: {np.sum(cs_values > fa_values)} runs")
    print(f"  Ties:    {np.sum(fa_values == cs_values)} runs")


def generate_all_visualizations(rosenbrock_results, knapsack_results, output_dir='results'):
    """Generate all visualizations for both problems."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"GENERATING VISUALIZATIONS")
    print(f"{'='*70}\n")
    
    # Rosenbrock Function Visualizations
    print("Rosenbrock Function Plots:")
    
    # 1. Surface plot
    plot_rosenbrock_surface(
        n_points=100,
        bounds=(-2, 2),
        save_path=f'{output_dir}/rosenbrock_surface.png',
        show=False
    )
    print("  ✓ rosenbrock_surface.png")
    
    # 2. Best run convergence
    fa_idx = np.argmin(rosenbrock_results['FA']['fitness'])
    cs_idx = np.argmin(rosenbrock_results['CS']['fitness'])
    plot_convergence(
        [rosenbrock_results['FA']['histories'][fa_idx],
         rosenbrock_results['CS']['histories'][cs_idx]],
        ['Firefly Algorithm', 'Cuckoo Search'],
        'Rosenbrock Function - Best Run Convergence',
        f'{output_dir}/rosenbrock_convergence_best.png',
        False
    )
    print("  ✓ rosenbrock_convergence_best.png")
    
    # 3. All runs with confidence intervals
    plot_multiple_runs(
        {'FA': rosenbrock_results['FA']['histories'],
         'CS': rosenbrock_results['CS']['histories']},
        ['FA', 'CS'],
        'Rosenbrock Function - All Runs (Mean ± Std)',
        f'{output_dir}/rosenbrock_convergence_all.png',
        False
    )
    print("  ✓ rosenbrock_convergence_all.png")
    
    # 4. Box plot
    plot_comparison(
        {'FA': rosenbrock_results['FA']['fitness'],
         'CS': rosenbrock_results['CS']['fitness']},
        'distribution',
        'Rosenbrock Function - Distribution',
        f'{output_dir}/rosenbrock_boxplot.png',
        False
    )
    print("  ✓ rosenbrock_boxplot.png")
    
    # 5. Statistics table
    plot_statistics_table(
        {'FA': rosenbrock_results['FA']['fitness'],
         'CS': rosenbrock_results['CS']['fitness']},
        f'{output_dir}/rosenbrock_statistics.png',
        False
    )
    print("  ✓ rosenbrock_statistics.png")
    
    # Knapsack Problem Visualizations
    print("\nKnapsack Problem Plots:")
    
    # 6. Best run convergence
    fa_idx = np.argmax(knapsack_results['FA']['values'])
    cs_idx = np.argmax(knapsack_results['CS']['values'])
    plot_convergence(
        [knapsack_results['FA']['histories'][fa_idx],
         knapsack_results['CS']['histories'][cs_idx]],
        ['Firefly Algorithm', 'Cuckoo Search'],
        'Knapsack Problem - Best Run Convergence',
        f'{output_dir}/knapsack_convergence_best.png',
        False
    )
    print("  ✓ knapsack_convergence_best.png")
    
    # 7. All runs with confidence intervals
    plot_multiple_runs(
        {'FA': knapsack_results['FA']['histories'],
         'CS': knapsack_results['CS']['histories']},
        ['FA', 'CS'],
        'Knapsack Problem - All Runs (Mean ± Std)',
        f'{output_dir}/knapsack_convergence_all.png',
        False
    )
    print("  ✓ knapsack_convergence_all.png")
    
    # 8. Box plot (using values, not fitness)
    plot_comparison(
        {'FA': knapsack_results['FA']['values'],
         'CS': knapsack_results['CS']['values']},
        'distribution',
        'Knapsack Problem - Value Distribution',
        f'{output_dir}/knapsack_boxplot.png',
        False
    )
    print("  ✓ knapsack_boxplot.png")
    
    # 9. Statistics table
    plot_statistics_table(
        {'FA': knapsack_results['FA']['values'],
         'CS': knapsack_results['CS']['values']},
        f'{output_dir}/knapsack_statistics.png',
        False
    )
    print("  ✓ knapsack_statistics.png")
    
    print(f"\n  All plots saved to: {os.path.abspath(output_dir)}/")


def main():
    """Main function to run all experiments."""
    print(f"\n{'#'*70}")
    print(f"# SWARM INTELLIGENCE ALGORITHMS - COMPREHENSIVE TESTING")
    print(f"# Algorithms: Firefly Algorithm (FA) & Cuckoo Search (CS)")
    print(f"# Problems: Rosenbrock Function & Knapsack Problem")
    print(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")
    
    start_time = time.time()
    
    # Configuration
    N_RUNS = 30
    ROSENBROCK_DIM = 10
    KNAPSACK_ITEMS = 50
    MAX_ITER = 500
    OUTPUT_DIR = 'results'
    
    # Run experiments
    rosenbrock_results = run_rosenbrock_experiments(
        n_runs=N_RUNS,
        n_dim=ROSENBROCK_DIM,
        max_iter=MAX_ITER
    )
    
    knapsack_results, _ = run_knapsack_experiments(
        n_runs=N_RUNS,
        n_items=KNAPSACK_ITEMS,
        max_iter=MAX_ITER,
        seed=42
    )
    
    # Print summary
    print_summary(rosenbrock_results, knapsack_results)
    
    # Generate visualizations
    generate_all_visualizations(rosenbrock_results, knapsack_results, OUTPUT_DIR)
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETED!")
    print(f"Total Execution Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Results saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
