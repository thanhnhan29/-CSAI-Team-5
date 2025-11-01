import numpy as np


def run_abc(fitness_function, bounds, n_solutions, n_dimensions, limit, max_iter):
    """
    Run the Artificial Bee Colony (ABC) algorithm.

    Args:
        fitness_function: Function to compute fitness for a solution.
        bounds: Lower and upper bounds for the search space, e.g., [(-20, 20), (-20, 20)].
        n_solutions (SN): Number of solutions in the population.
        n_dimensions (n): Number of dimensions in the problem.
        limit: Abandonment threshold for Scout Bees.
        max_iter: Maximum number of iterations.

    Returns:
        tuple: (best_solution, best_fitness)
            - best_solution: The best solution found
            - best_fitness: The fitness value of the best solution
    """
    
    # Extract lower bounds (lb) and upper bounds (ub)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    # --- STEP 1: INITIALIZATION ---
    # Create initial population (Scout Bee formula)
    population = np.zeros((n_solutions, n_dimensions))
    for i in range(n_solutions):
        # Formula 3: x_ik = lb_k + rand(0,1) * (ub_k - lb_k)
        population[i] = lb + np.random.rand(n_dimensions) * (ub - lb)
    
    # Calculate fitness for initial population
    fitness_values = np.array([fitness_function(sol) for sol in population])
    
    # Initialize trial counter (abandonment counter)
    trials_counter = np.zeros(n_solutions)
    
    # Remember the best initial solution
    best_solution = population[np.argmax(fitness_values)].copy()
    best_fitness = np.max(fitness_values)
    
    print(f"Iteration 0 | Best fitness: {best_fitness:.4f}")

    # --- MAIN LOOP ---
    for iteration in range(max_iter):
        
        # --- PHASE 1: EMPLOYED BEES ---
        for i in range(n_solutions):
            # Choose a random neighbor j different from i
            j_list = list(range(n_solutions))
            j_list.remove(i)
            j = np.random.choice(j_list)
            
            # Choose a random dimension k
            k = np.random.randint(0, n_dimensions)
            
            # Get a random number Phi in [-1, 1]
            phi = np.random.uniform(-1, 1)
            
            # Create candidate solution
            candidate_solution = population[i].copy()
            
            # Apply Formula: v_ik = x_ik + Phi * (x_ik - x_jk), choosing the direction to which the result should vary
            candidate_solution[k] = population[i, k] + phi * (population[i, k] - population[j, k])
            
            # Ensure the new value doesn't exceed bounds
            candidate_solution[k] = np.clip(candidate_solution[k], lb[k], ub[k])
            
            # Evaluate candidate solution
            candidate_fitness = fitness_function(candidate_solution)
            
            # Greedy selection
            if candidate_fitness > fitness_values[i]:
                population[i] = candidate_solution
                fitness_values[i] = candidate_fitness
                trials_counter[i] = 0
            else:
                trials_counter[i] += 1

        # --- PHASE 2: ONLOOKER BEES ---
        # Apply Formula : P_i = fit_i / sum(fit_j)
        # To avoid negative probabilities, shift fitness values to be non-negative
        min_fitness = np.min(fitness_values)
        if min_fitness < 0:
            shifted_fitness = fitness_values - min_fitness  # Shift to >= 0
        else:
            shifted_fitness = fitness_values
        
        total_fitness = np.sum(shifted_fitness)
        if total_fitness == 0:
            probabilities = np.ones(n_solutions) / n_solutions  # Equal probability
        else:
            probabilities = shifted_fitness / total_fitness
            
        # Onlooker bees loop
        for _ in range(n_solutions):  # There are SN onlooker bees
            # Select a solution based on probability (roulette wheel)
            selected_index = np.random.choice(range(n_solutions), p=probabilities)
            
            # ----- ONLOOKER BEE ACTION -----
            # (Repeat the exact logic of Employed Bee for the selected solution)
            
            # Choose random neighbor j different from selected_index
            j_list = list(range(n_solutions))
            j_list.remove(selected_index)
            j = np.random.choice(j_list)
            k = np.random.randint(0, n_dimensions)
            phi = np.random.uniform(-1, 1)
            
            candidate_solution = population[selected_index].copy()
            # Apply Formula 1 again
            candidate_solution[k] = population[selected_index, k] + phi * (population[selected_index, k] - population[j, k])
            candidate_solution[k] = np.clip(candidate_solution[k], lb[k], ub[k])
            
            candidate_fitness = fitness_function(candidate_solution)
            
            if candidate_fitness > fitness_values[selected_index]:
                population[selected_index] = candidate_solution
                fitness_values[selected_index] = candidate_fitness
                trials_counter[selected_index] = 0
            else:
                trials_counter[selected_index] += 1
                
        # --- PHASE 3: SCOUT BEES ---
        for i in range(n_solutions):
            if trials_counter[i] > limit:
                # Replace stuck solution with a new random solution
                # Apply Formula 
                population[i] = lb + np.random.rand(n_dimensions) * (ub - lb)
                fitness_values[i] = fitness_function(population[i])
                trials_counter[i] = 0
        
        current_best_fitness = np.max(fitness_values)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[np.argmax(fitness_values)].copy()
        
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1} | Best fitness: {best_fitness:.4f}")
            
    return best_solution, best_fitness
