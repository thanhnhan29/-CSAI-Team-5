import numpy as np


def steepest_ascent_hill_climbing(fitness_function, bounds, n_dimensions, epsilon, n_neighbors, run_limit):
    """
    Run the Steepest Ascent Hill Climbing algorithm.

    Args:
        fitness_function: Function to compute fitness for a solution.
        bounds: Lower and upper bounds for the search space.
        n_dimensions (n): Number of dimensions in the problem.
        epsilon: Step size for creating neighbors.
        n_neighbors: Number of neighbors generated at each step.
        run_limit: Maximum number of jumps allowed.

    Returns:
        tuple: (best_solution, best_fitness)
            - best_solution: The best solution found
            - best_fitness: The fitness value of the best solution
    """
    
    # Extract lower bounds (lb) and upper bounds (ub)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    # --- STEP 1: INITIALIZATION ---
    # Initialize a single random solution
    current_solution = lb + np.random.rand(n_dimensions) * (ub - lb)
    current_fitness = fitness_function(current_solution)
    
    print(f"Starting point at {np.round(current_solution, 2)} | Fitness: {current_fitness:.4f}")

    # --- MAIN LOOP ---
    # Repeat until the jump limit is reached
    for jump_count in range(run_limit):
        
        # --- GENERATE AND EVALUATE ALL NEIGHBORS ---
        best_neighbor = None
        # Initialize best neighbor fitness with current fitness
        best_neighbor_fitness = current_fitness 

        for _ in range(n_neighbors):
            # Create a new neighbor
            # Generate random step in the range [-epsilon, epsilon] for all dimensions
            step = np.random.uniform(-epsilon, epsilon, n_dimensions)
            candidate_neighbor = current_solution + step
            
            # Ensure neighbor doesn't exceed bounds
            candidate_neighbor = np.clip(candidate_neighbor, lb, ub)
            
            # Evaluate neighbor
            neighbor_fitness = fitness_function(candidate_neighbor)
            
            # Compare with the BEST neighbor found IN THIS ROUND
            if neighbor_fitness > best_neighbor_fitness:
                best_neighbor_fitness = neighbor_fitness
                best_neighbor = candidate_neighbor

        # --- MAKE MOVE DECISION ---
        # Compare best neighbor with CURRENT position
        if best_neighbor is not None and best_neighbor_fitness > current_fitness:
            # Move to better position
            current_solution = best_neighbor
            current_fitness = best_neighbor_fitness
            print(f"  Jump {jump_count + 1}: moved to {np.round(current_solution, 2)} | New fitness: {current_fitness:.4f}")
        else:
            # Stuck! No better neighbor found.
            print(f"\nStuck at iteration {jump_count + 1}. No better neighbor found.")
            break  # Exit main loop

    print(f"\nAlgorithm stopped after {jump_count + 1} checks.")
    return current_solution, current_fitness
