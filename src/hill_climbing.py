import numpy as np

def steepest_ascent_hill_climbing(fitness_function, bounds, n_dimensions, epsilon, n_neighbors, run_limit):
    """
    Hàm chạy thuật toán Steepest Ascent Hill Climbing.

    Args:
        fitness_function: Hàm tính fitness cho một lời giải.
        bounds: Biên dưới và biên trên của không gian tìm kiếm.
        n_dimensions (n): Số chiều của bài toán.
        epsilon: Kích thước bước nhảy để tạo hàng xóm.
        n_neighbors: Số lượng hàng xóm được tạo ra ở mỗi bước.
        run_limit: Số lần nhảy tối đa.

    Returns:
        Lời giải tốt nhất và fitness tương ứng.
    """
    
    # Tách biên dưới (lb) và biên trên (ub)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    # --- BƯỚC 1: KHỞI TẠO ---
    # Khởi tạo một lời giải duy nhất ngẫu nhiên
    current_solution = lb + np.random.rand(n_dimensions) * (ub - lb)
    current_fitness = fitness_function(current_solution)
    
    print(f"Starting point at {np.round(current_solution, 2)} | Fitness: {current_fitness:.4f}")

    # --- VÒNG LẶP CHÍNH ---
    # Lặp lại cho đến khi đạt giới hạn số lần nhảy
    for jump_count in range(run_limit):
        
        # --- TẠO VÀ ĐÁNH GIÁ TOÀN BỘ HÀNG XÓM ---
        best_neighbor = None
        # Khởi tạo fitness của hàng xóm tốt nhất bằng fitness hiện tại
        best_neighbor_fitness = current_fitness 

        for _ in range(n_neighbors):
            # Tạo một hàng xóm mới
            # Tạo bước nhảy ngẫu nhiên trong khoảng [-epsilon, epsilon] cho tất cả các chiều
            step = np.random.uniform(-epsilon, epsilon, n_dimensions)
            candidate_neighbor = current_solution + step
            
            # Đảm bảo hàng xóm không vượt ra ngoài biên
            candidate_neighbor = np.clip(candidate_neighbor, lb, ub)
            
            # Đánh giá hàng xóm
            neighbor_fitness = fitness_function(candidate_neighbor)
            
            # So sánh với hàng xóm TỐT NHẤT đã tìm thấy TRONG VÒNG NÀY
            if neighbor_fitness > best_neighbor_fitness:
                best_neighbor_fitness = neighbor_fitness
                best_neighbor = candidate_neighbor

        # --- ĐƯA RA QUYẾT ĐỊNH DI CHUYỂN ---
        # So sánh hàng xóm tốt nhất với vị trí HIỆN TẠI
            if best_neighbor is not None and best_neighbor_fitness > current_fitness:
                # Di chuyển đến vị trí tốt hơn
                current_solution = best_neighbor
                current_fitness = best_neighbor_fitness
                print(f"  Jump {jump_count + 1}: moved to {np.round(current_solution, 2)} | New fitness: {current_fitness:.4f}")
            else:
                # Bị kẹt! Không có hàng xóm nào tốt hơn.
                print(f"\nStuck at iteration {jump_count + 1}. No better neighbor found.")
                break # Thoát khỏi vòng lặp chính

    print(f"\nAlgorithm stopped after {jump_count + 1} checks.")
    return current_solution, current_fitness


# --- CÁCH CHẠY VÀ KẾT QUẢ ---
if __name__ == "__main__":
    # 1. Định nghĩa hàm fitness cho bài toán (giống hệt bài ABC)
    def objective_function(v):
        x, y = v
        # f(x, y) = 100 - ((x - 10)^2 + (y + 5)^2)
        return 100 - (np.power(x - 10, 2) + np.power(y + 5, 2))

    # 2. Thiết lập các tham số
    bounds = [(-20, 20), (-20, 20)]  # Biên cho x và y
    N_DIM = 2                       # Số chiều
    EPSILON = 0.5                   # Kích thước bước nhảy
    N_NEIGHBORS = 20                # Mỗi lần tạo 20 hàng xóm để kiểm tra
    RUN_LIMIT = 100                  # Giới hạn 100 lần nhảy thành công

    # 3. Chạy thuật toán Hill Climbing
    print("Starting Steepest Ascent Hill Climbing...")
    solution, fitness = steepest_ascent_hill_climbing(objective_function, bounds, N_DIM, EPSILON, N_NEIGHBORS, RUN_LIMIT)

    # 4. In kết quả
    print("\n--- FINISHED ---")
    print("Best solution found at:")
    print(f"  x = {solution[0]:.4f}")
    print(f"  y = {solution[1]:.4f}")
    print(f"With best fitness value: {fitness:.4f}")
    print("(Theoretical answer is x=10, y=-5, fitness=100)")