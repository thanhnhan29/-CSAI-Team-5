"""
Main entry point for running optimization algorithms.
Demonstrates ABC and Hill Climbing algorithms.
Uses Hydra for configuration management.
"""
import numpy as np
import hydra
from omegaconf import DictConfig
from algorithms.abc.abc import run_abc
from algorithms.hill_climbing.hill_climbing import steepest_ascent_hill_climbing
from Problem.RF.RF import RF


def objective_function(v, optim_max = False):
    """
    Objective function for optimization.
    f(x, y) = 100 - ((x - 10)^2 + (y + 5)^2)
    Maximum at (10, -5) with value 100.
    """
    x, y = v
    rf = RF(a=4, b=10000)
    return rf(x, y) if optim_max else -rf(x, y)
    #return 100 - ((x - 10) ** 2 + (y + 5) ** 2)


def run_abc_demo(cfg: DictConfig):
    """Run ABC algorithm demo"""
    print("\n" + "="*60)
    print("Algorithm 1: Artificial Bee Colony (ABC)")
    print("="*60)
    
    # Load parameters from config
    bounds = [tuple(b) for b in cfg.common.bounds]
    n_dim = cfg.common.n_dimensions
    n_solutions = cfg.abc.n_solutions
    limit = cfg.abc.limit
    max_iter = cfg.abc.max_iter
    
    # Hardcoded version (kept for reference):
    # bounds = [(-20, 20), (-20, 20)]
    # n_dim = 2
    # n_solutions = 50
    # limit = 100
    # max_iter = 200

    print("Starting ABC...")
    print(f"Parameters: n_solutions={n_solutions}, limit={limit}, max_iter={max_iter}")
    solution, fitness = run_abc(objective_function, bounds, n_solutions, n_dim, limit, max_iter)

    print("\n--- ABC RESULTS ---")
    print(f"Best solution found at:")
    print(f"  x = {solution[0]:.4f}")
    print(f"  y = {solution[1]:.4f}")
    print(f"With maximum fitness value: {fitness:.4f}")


def run_hill_climbing_demo(cfg: DictConfig):
    """Run Hill Climbing algorithm demo with Hydra config"""
    print("\n" + "="*60)
    print("Algorithm 2: Steepest Ascent Hill Climbing")
    print("="*60)
    
    # Load parameters from config
    bounds = [tuple(b) for b in cfg.common.bounds]
    n_dim = cfg.common.n_dimensions
    epsilon = cfg.hill_climbing.epsilon
    n_neighbors = cfg.hill_climbing.n_neighbors
    run_limit = cfg.hill_climbing.run_limit
    
    # Hardcoded version (kept for reference):
    # bounds = [(-20, 20), (-20, 20)]
    # n_dim = 2
    # epsilon = 0.5
    # n_neighbors = 20
    # run_limit = 100

    print("Starting Steepest Ascent Hill Climbing...")
    print(f"Parameters: epsilon={epsilon}, n_neighbors={n_neighbors}, run_limit={run_limit}")
    solution, fitness = steepest_ascent_hill_climbing(
        objective_function, bounds, n_dim, epsilon, n_neighbors, run_limit
    )

    print("\n--- HILL CLIMBING RESULTS ---")
    print("Best solution found at:")
    print(f"  x = {solution[0]:.4f}")
    print(f"  y = {solution[1]:.4f}")
    print(f"With best fitness value: {fitness:.4f}")


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point with Hydra configuration.
    
    Usage:
        python main.py                                    # Use default config
        python main.py abc.n_solutions=100               # Override ABC population size
        python main.py hill_climbing.epsilon=1.0         # Override hill climbing step size
    """
    print("="*60)
    print("Optimization Algorithms Framework Demo")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Bounds: {cfg.common.bounds}")
    print(f"  ABC: n_solutions={cfg.abc.n_solutions}, max_iter={cfg.abc.max_iter}")
    print(f"  Hill Climbing: epsilon={cfg.hill_climbing.epsilon}, n_neighbors={cfg.hill_climbing.n_neighbors}")
    
    run_abc_demo(cfg)
    run_hill_climbing_demo(cfg)
    
    print("\n" + "="*60)
    print("All demos completed!")
    print("="*60)


if __name__ == "__main__":
    main()
