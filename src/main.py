import numpy as np
from aco.colony import ACO
from pso.swarm import PSO
from search.graph_search import GraphSearch
from Problem.KP.kp import KP
from Problem.RF.RF import RF

def run_knapsack_aco():
    """Run ACO for Knapsack Problem"""
    num_items = 500
    capacity = 5000
    kp = KP(num=num_items, capacity=capacity, nmax=100, nmin=1, seed=42)
    kp.set_init()
    
    print(f"Knapsack Problem: {num_items} items, capacity = {capacity}")
    print(f"Weights: {kp.weights[:10]}... (showing first 10)")
    print(f"Values: {kp.values[:10]}... (showing first 10)")
    
    optimal_value = kp.solve()
    print(f"\nOptimal value (DP): {optimal_value}")
    
    dist_matrix = np.zeros((num_items, num_items))
    for i in range(num_items):
        for j in range(num_items):
            if i != j:
                ratio_i = kp.values[i] / (kp.weights[i] + 1e-10)
                ratio_j = kp.values[j] / (kp.weights[j] + 1e-10)
                dist_matrix[i][j] = 1.0 / (abs(ratio_i - ratio_j) + 0.1)
            else:
                dist_matrix[i][j] = 1e-10
    
    def compute_knapsack_reward(route):
        total_weight = 0
        total_value = 0
        for item_idx in route:
            if total_weight + kp.weights[item_idx] <= capacity:
                total_weight += kp.weights[item_idx]
                total_value += kp.values[item_idx]
        return total_value if total_weight <= capacity else 0
    
    def knapsack_stop_condition(route):
        if len(route) >= num_items:
            return False
        
        current_weight = sum(kp.weights[i] for i in route)
        if current_weight >= capacity:
            return False
        
        remaining_capacity = capacity - current_weight
        can_add = any(kp.weights[i] <= remaining_capacity for i in range(num_items) if i not in route)
        return can_add and len(route) < num_items
    
    aco = ACO(
        num_ants=30,
        num_iter=100,
        alpha=1.0,
        beta=3.0,
        rho=0.5,
        q=100,
        dist_matrix=dist_matrix,
        compute_reward=compute_knapsack_reward,
        stop_condition=knapsack_stop_condition
    )

    best_route, best_value = aco.run()

    print("\n✅ Best route (selected items):", best_route[:20], "..." if len(best_route) > 20 else "")
    print(f"🏁 Best value: {best_value:.4f}")
    print(f"📊 Gap from optimal: {((optimal_value - best_value) / optimal_value * 100):.2f}%")
    
    total_weight = sum(kp.weights[i] for i in best_route)
    print(f"⚖️  Total weight: {total_weight}/{capacity}")

def run_tsp_aco():
    """Run ACO for TSP (default behavior)"""
    num_cities = 20
    
    np.random.seed(42)
    cities = np.random.rand(num_cities, 2) * 100
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            dist_matrix[i][j] = np.linalg.norm(cities[i] - cities[j])
    
    print(f"🗺️  TSP Problem: {num_cities} cities")

    aco = ACO(
        num_ants=20,
        num_iter=100,
        alpha=1.0,
        beta=5.0,
        rho=0.5,
        q=100,
        dist_matrix=dist_matrix
    )

    best_route, best_length = aco.run()

    print("\n✅ Best route:", best_route)
    print(f"🏁 Best length: {-best_length:.4f}")

def run_knapsack_pso():
    """Run PSO for Knapsack Problem"""
    num_items = 500
    capacity = 5000
    kp = KP(num=num_items, capacity=capacity, nmax=100, nmin=1, seed=42)
    kp.set_init()
    
    print(f"Knapsack Problem: {num_items} items, capacity = {capacity}")
    print(f"Weights: {kp.weights[:10]}... (showing first 10)")
    print(f"Values: {kp.values[:10]}... (showing first 10)")
    
    optimal_value = kp.solve()
    print(f"\nOptimal value (DP): {optimal_value}")
    
    def repair_solution(solution):
        """Repair function to ensure feasibility"""
        solution_copy = solution.copy().astype(int)
        total_weight = np.sum(kp.weights * solution_copy)
        
        if total_weight <= capacity:
            return solution_copy
        
        selected_indices = np.where(solution_copy == 1)[0]
        if len(selected_indices) == 0:
            return solution_copy
        
        # Remove items with worst value/weight ratio
        ratios = [(idx, kp.values[idx] / kp.weights[idx]) for idx in selected_indices]
        ratios.sort(key=lambda x: x[1])
        
        for idx, _ in ratios:
            if total_weight <= capacity:
                break
            solution_copy[idx] = 0
            total_weight -= kp.weights[idx]
        
        return solution_copy
    
    def compute_knapsack_fitness(solution):
        """Fitness function for knapsack"""
        solution_binary = solution.astype(int)
        total_weight = np.sum(kp.weights * solution_binary)
        total_value = np.sum(kp.values * solution_binary)
        
        if total_weight > capacity:
            penalty = (total_weight - capacity) / capacity
            return -penalty
        return total_value
    
    bounds = [(0, 1) for _ in range(num_items)]
    
    pso = PSO(
        num_particles=30,
        num_iter=100,
        num_dimensions=num_items,
        bounds=bounds,
        compute_fitness=compute_knapsack_fitness,
        particle_type='binary',
        w=0.9,
        c1=2.0,
        c2=2.0,
        w_decay=0.95,
        repair_func=repair_solution
    )
    
    best_solution, best_value = pso.run()
    
    best_solution_binary = best_solution.astype(int)
    total_weight = np.sum(kp.weights * best_solution_binary)
    selected_items = np.where(best_solution_binary == 1)[0]
    
    print("\n✅ Best solution (selected items):", selected_items[:20], "..." if len(selected_items) > 20 else "")
    print(f"🏁 Best value: {best_value:.4f}")
    print(f"📊 Gap from optimal: {((optimal_value - best_value) / optimal_value * 100):.2f}%")
    print(f"⚖️  Total weight: {total_weight}/{capacity}")

def run_rosenbrock_pso():
    """Run PSO for Rosenbrock Function (RF) optimization"""
    a = 1
    b = 100
    rf = RF(a, b)
    
    print(f"Rosenbrock Function: a={a}, b={b}")
    x_min, y_min, f_min = rf.global_minimum()
    print(f"Global minimum: f({x_min}, {y_min}) = {f_min}")
    
    def compute_rf_fitness(position):
        """Fitness function for Rosenbrock"""
        x, y = position[0], position[1]
        return -rf(x, y)
    
    bounds = [(-5, 5), (-5, 5)]
    
    pso = PSO(
        num_particles=30,
        num_iter=200,
        num_dimensions=2,
        bounds=bounds,
        compute_fitness=compute_rf_fitness,
        particle_type='continuous',
        w=0.7,
        c1=1.5,
        c2=1.5,
        w_decay=0.99
    )
    
    best_position, best_fitness = pso.run()
    best_x, best_y = best_position[0], best_position[1]
    actual_value = rf(best_x, best_y)
    
    print(f"\n✅ Best position: x={best_x:.6f}, y={best_y:.6f}")
    print(f"🏁 Function value: f(x, y) = {actual_value:.6f}")
    print(f"📊 Distance from global minimum: {np.sqrt((best_x - x_min)**2 + (best_y - y_min)**2):.6f}")
    print(f"🎯 Error from optimal value: {abs(actual_value - f_min):.6e}")

def run_knapsack_bfs():
    """Run BFS for Knapsack Problem (small instance)"""
    num_items = 15
    capacity = 50
    kp = KP(num=num_items, capacity=capacity, nmax=20, nmin=1, seed=42)
    kp.set_init()
    
    print(f"Knapsack Problem: {num_items} items, capacity = {capacity}")
    print(f"Weights: {kp.weights}")
    print(f"Values: {kp.values}")
    
    optimal_value = kp.solve()
    print(f"\nOptimal value (DP): {optimal_value}")
    
    initial_state = (0, 0, 0, ())
    
    def goal_test(state):
        _, _, items_considered, _ = state
        return items_considered == num_items
    
    def get_successors(state):
        current_weight, current_value, items_considered, selected_items = state
        if items_considered >= num_items:
            return []
        
        successors = []
        item_idx = items_considered
        
        # Skip item
        next_state = (current_weight, current_value, items_considered + 1, selected_items)
        successors.append(('skip', next_state, 0))
        
        # Take item (if fits)
        if current_weight + kp.weights[item_idx] <= capacity:
            new_weight = current_weight + kp.weights[item_idx]
            new_value = current_value + kp.values[item_idx]
            new_selected = selected_items + (item_idx,)
            next_state = (new_weight, new_value, items_considered + 1, new_selected)
            successors.append(('take', next_state, -kp.values[item_idx]))
        
        return successors
    
    search = GraphSearch(initial_state, goal_test, get_successors)
    result = search.bfs()
    
    if result:
        final_weight, final_value, _, selected_items = result.state
        print(f"\n✅ Best solution: {selected_items}")
        print(f"🏁 Total value: {final_value}")
        print(f"📊 Gap from optimal: {((optimal_value - final_value) / optimal_value * 100):.2f}%")
        print(f"⚖️  Total weight: {final_weight}/{capacity}")

def run_knapsack_dfs():
    """Run DFS for Knapsack Problem (small instance)"""
    num_items = 15
    capacity = 50
    kp = KP(num=num_items, capacity=capacity, nmax=20, nmin=1, seed=42)
    kp.set_init()
    
    print(f"Knapsack Problem: {num_items} items, capacity = {capacity}")
    print(f"Weights: {kp.weights}")
    print(f"Values: {kp.values}")
    
    optimal_value = kp.solve()
    print(f"\nOptimal value (DP): {optimal_value}")
    
    initial_state = (0, 0, 0, ())
    
    def goal_test(state):
        _, _, items_considered, _ = state
        return items_considered == num_items
    
    def get_successors(state):
        current_weight, current_value, items_considered, selected_items = state
        if items_considered >= num_items:
            return []
        
        successors = []
        item_idx = items_considered
        
        # Skip item
        next_state = (current_weight, current_value, items_considered + 1, selected_items)
        successors.append(('skip', next_state, 0))
        
        # Take item (if fits)
        if current_weight + kp.weights[item_idx] <= capacity:
            new_weight = current_weight + kp.weights[item_idx]
            new_value = current_value + kp.values[item_idx]
            new_selected = selected_items + (item_idx,)
            next_state = (new_weight, new_value, items_considered + 1, new_selected)
            successors.append(('take', next_state, -kp.values[item_idx]))
        
        return successors
    
    search = GraphSearch(initial_state, goal_test, get_successors)
    result = search.dfs(max_depth=num_items * 2)
    
    if result:
        final_weight, final_value, _, selected_items = result.state
        print(f"\n✅ Best solution: {selected_items}")
        print(f"🏁 Total value: {final_value}")
        print(f"📊 Gap from optimal: {((optimal_value - final_value) / optimal_value * 100):.2f}%")
        print(f"⚖️  Total weight: {final_weight}/{capacity}")

def run_knapsack_astar():
    """Run A* for Knapsack Problem (small instance)"""
    num_items = 15
    capacity = 50
    kp = KP(num=num_items, capacity=capacity, nmax=20, nmin=1, seed=42)
    kp.set_init()
    
    print(f"Knapsack Problem: {num_items} items, capacity = {capacity}")
    print(f"Weights: {kp.weights}")
    print(f"Values: {kp.values}")
    
    optimal_value = kp.solve()
    print(f"\nOptimal value (DP): {optimal_value}")
    
    initial_state = (0, 0, 0, ())
    
    def goal_test(state):
        _, _, items_considered, _ = state
        return items_considered == num_items
    
    def get_successors(state):
        current_weight, current_value, items_considered, selected_items = state
        if items_considered >= num_items:
            return []
        
        successors = []
        item_idx = items_considered
        
        # Skip item
        next_state = (current_weight, current_value, items_considered + 1, selected_items)
        successors.append(('skip', next_state, 0))
        
        # Take item (if fits)
        if current_weight + kp.weights[item_idx] <= capacity:
            new_weight = current_weight + kp.weights[item_idx]
            new_value = current_value + kp.values[item_idx]
            new_selected = selected_items + (item_idx,)
            next_state = (new_weight, new_value, items_considered + 1, new_selected)
            successors.append(('take', next_state, -kp.values[item_idx]))
        
        return successors
    
    def heuristic(state):
        """Optimistic estimate of remaining value"""
        current_weight, _, items_considered, _ = state
        remaining_capacity = capacity - current_weight
        
        remaining_items = []
        for i in range(items_considered, num_items):
            if kp.weights[i] <= remaining_capacity:
                ratio = kp.values[i] / kp.weights[i]
                remaining_items.append((ratio, kp.values[i], kp.weights[i]))
        
        remaining_items.sort(reverse=True)
        
        estimated_value = 0
        estimated_weight = 0
        for ratio, value, weight in remaining_items:
            if estimated_weight + weight <= remaining_capacity:
                estimated_value += value
                estimated_weight += weight
        
        return -estimated_value
    
    search = GraphSearch(initial_state, goal_test, get_successors, heuristic)
    result = search.a_star()
    
    if result:
        final_weight, final_value, _, selected_items = result.state
        print(f"\n✅ Best solution: {selected_items}")
        print(f"🏁 Total value: {final_value}")
        print(f"📊 Gap from optimal: {((optimal_value - final_value) / optimal_value * 100):.2f}%")
        print(f"⚖️  Total weight: {final_weight}/{capacity}")

if __name__ == "__main__":
    print("="*60)
    print("🐜 ACO Framework Demo")
    print("="*60)
    
    print("\n" + "="*60)
    print("Problem 1: Knapsack Problem - ACO")
    print("="*60)
    run_knapsack_aco()
    
    print("\n\n" + "="*60)
    print("Problem 2: TSP (Traveling Salesman Problem) - ACO")
    print("="*60)
    run_tsp_aco()
    
    print("\n\n" + "="*60)
    print("="*60)
    print("🌊 PSO Framework Demo")
    print("="*60)
    
    print("\n" + "="*60)
    print("Problem 1: Knapsack Problem - PSO")
    print("="*60)
    run_knapsack_pso()
    
    print("\n\n" + "="*60)
    print("Problem 2: Rosenbrock Function - PSO")
    print("="*60)
    run_rosenbrock_pso()
    
    print("\n\n" + "="*60)
    print("="*60)
    print("🔍 Graph Search Algorithms Demo")
    print("="*60)
    
    print("\n" + "="*60)
    print("Problem: Knapsack (Small) - BFS")
    print("="*60)
    run_knapsack_bfs()
    
    print("\n\n" + "="*60)
    print("Problem: Knapsack (Small) - DFS")
    print("="*60)
    run_knapsack_dfs()
    
    print("\n\n" + "="*60)
    print("Problem: Knapsack (Small) - A*")
    print("="*60)
    run_knapsack_astar()
