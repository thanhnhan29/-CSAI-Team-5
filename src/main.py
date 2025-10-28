import numpy as np
from aco.colony import ACO
from Problem.KP.kp import KP

def run_knapsack_aco():
    """Run ACO for Knapsack Problem"""
    num_items = 500
    capacity = 5000
    kp = KP(num=num_items, capacity=capacity, nmax=100, nmin=1, seed=42)
    kp.set_init()
    
    print(f"📦 Knapsack Problem: {num_items} items, capacity = {capacity}")
    print(f"Weights: {kp.weights[:10]}... (showing first 10)")
    print(f"Values: {kp.values[:10]}... (showing first 10)")
    
    optimal_value = kp.solve()
    print(f"\n🎯 Optimal value (DP): {optimal_value}")
    
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

if __name__ == "__main__":
    print("="*60)
    print("🐜 ACO Framework Demo")
    print("="*60)
    
    print("\n" + "="*60)
    print("Problem 1: Knapsack Problem")
    print("="*60)
    run_knapsack_aco()
    
    print("\n\n" + "="*60)
    print("Problem 2: TSP (Traveling Salesman Problem)")
    print("="*60)
    run_tsp_aco()
