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
    pass

def run_rosenbrock_pso():
    """Run PSO for Rosenbrock Function (RF) optimization"""
    pass

def run_knapsack_search(algorithm='bfs', max_items=15):
    """
    Run BFS/DFS/A* for Knapsack Problem
    Note: Limited to fewer items due to exponential state space
    """
    capacity = 50
    kp = KP(num=max_items, capacity=capacity, nmax=20, nmin=1, seed=42)
    kp.set_init()
    
    print(f"Knapsack Problem: {max_items} items, capacity = {capacity}")
    print(f"Weights: {kp.weights}")
    print(f"Values: {kp.values}")
    
    optimal_value = kp.solve()
    print(f"\nOptimal value (DP): {optimal_value}")
    
    # State representation: tuple of (current_weight, current_value, items_considered, selected_items_tuple)
    initial_state = (0, 0, 0, ())
    
    def goal_test(state):
        """Goal: all items have been considered"""
        _, _, items_considered, _ = state
        return items_considered == max_items
    
    def get_successors(state):
        """Generate successor states: include or exclude next item"""
        current_weight, current_value, items_considered, selected_items = state
        
        if items_considered >= max_items:
            return []
        
        successors = []
        item_idx = items_considered
        
        # Option 1: Don't take the item
        next_state = (current_weight, current_value, items_considered + 1, selected_items)
        successors.append(('skip', next_state, 0))
        
        # Option 2: Take the item (if it fits)
        if current_weight + kp.weights[item_idx] <= capacity:
            new_weight = current_weight + kp.weights[item_idx]
            new_value = current_value + kp.values[item_idx]
            new_selected = selected_items + (item_idx,)
            next_state = (new_weight, new_value, items_considered + 1, new_selected)
            # Negative cost because we want to maximize value
            successors.append(('take', next_state, -kp.values[item_idx]))
        
        return successors
    
    def heuristic(state):
        """A* heuristic: optimistic estimate of remaining value"""
        current_weight, current_value, items_considered, _ = state
        remaining_capacity = capacity - current_weight
        
        # Upper bound: take items with best value/weight ratio that fit
        remaining_items = []
        for i in range(items_considered, max_items):
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
        
        # Return negative because we're minimizing cost but maximizing value
        return -estimated_value
    
    # Create search instance
    search = GraphSearch(initial_state, goal_test, get_successors, heuristic)
    
    # Run selected algorithm
    if algorithm == 'bfs':
        result = search.bfs()
    elif algorithm == 'dfs':
        result = search.dfs(max_depth=max_items * 2)
    elif algorithm == 'a_star':
        result = search.a_star()
    else:
        print(f"Unknown algorithm: {algorithm}")
        return
    
    if result:
        final_weight, final_value, _, selected_items = result.state
        print(f"\n✅ Best solution found: {selected_items}")
        print(f"🏁 Total value: {final_value}")
        print(f"📊 Gap from optimal: {((optimal_value - final_value) / optimal_value * 100):.2f}%")
        print(f"⚖️  Total weight: {final_weight}/{capacity}")
        print(f"📏 Solution depth: {result.depth}")

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
    run_knapsack_search(algorithm='bfs', max_items=15)
    
    print("\n\n" + "="*60)
    print("Problem: Knapsack (Small) - DFS")
    print("="*60)
    run_knapsack_search(algorithm='dfs', max_items=15)
    
    print("\n\n" + "="*60)
    print("Problem: Knapsack (Small) - A*")
    print("="*60)
    run_knapsack_search(algorithm='a_star', max_items=15)
