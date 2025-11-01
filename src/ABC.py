import numpy as np


def objective_function(v):
    x, y = v
    # f(x, y) = 100 - ((x - 10)^2 + (y + 5)^2)
    # Càng gần (10, -5) thì giá trị càng lớn
    return 100 - (np.power(x - 10, 2) + np.power(y + 5, 2))


def run_abc(fitness_function, bounds, n_solutions, n_dimensions, limit, max_iter):
    """
    Hàm chạy thuật toán ABC đơn giản dựa trên các công thức toán học.

    Args:
        fitness_function: Hàm tính fitness cho một lời giải.
        bounds: Biên dưới và biên trên của không gian tìm kiếm, ví dụ: [(-20, 20), (-20, 20)].
        n_solutions (SN): Số lượng lời giải trong quần thể.
        n_dimensions (n): Số chiều của bài toán.
        limit: Ngưỡng từ bỏ cho Ong Trinh sát.
        max_iter: Số vòng lặp tối đa.

    Returns:
        Lời giải tốt nhất và fitness tương ứng.
    """
    
    # Tách biên dưới (lb) và biên trên (ub)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    # --- BƯỚC 1: KHỞI TẠO ---
    # Tạo quần thể ban đầu bằng Công thức 3 (của Ong Trinh sát)
    population = np.zeros((n_solutions, n_dimensions))
    for i in range(n_solutions):
        # Công thức 3: x_ik = lb_k + rand(0,1) * (ub_k - lb_k)
        population[i] = lb + np.random.rand(n_dimensions) * (ub - lb)
    
    # Tính fitness cho quần thể ban đầu
    fitness_values = np.array([fitness_function(sol) for sol in population])
    
    # Khởi tạo bộ đếm từ bỏ (trials counter)
    trials_counter = np.zeros(n_solutions)
    
    # Ghi nhớ lời giải tốt nhất ban đầu
    best_solution = population[np.argmax(fitness_values)].copy()
    best_fitness = np.max(fitness_values)
    
    print(f"Loop 0 | best fitness: {best_fitness:.4f}")

    # --- VÒNG LẶP CHÍNH ---
    for iteration in range(max_iter):
        
        # --- GIAI ĐOẠN 1: ONG THỢ (EMPLOYED BEES) ---
        for i in range(n_solutions):
            # Chọn một "hàng xóm" ngẫu nhiên j khác i
            j_list = list(range(n_solutions))
            j_list.remove(i)
            j = np.random.choice(j_list)
            
            # Chọn một chiều ngẫu nhiên k
            k = np.random.randint(0, n_dimensions)
            
            # Lấy một số ngẫu nhiên Phi trong [-1, 1]
            phi = np.random.uniform(-1, 1)
            
            # Tạo lời giải ứng viên mới
            candidate_solution = population[i].copy()
            
            # Áp dụng Công thức 1: v_ik = x_ik + Phi * (x_ik - x_jk)
            candidate_solution[k] = population[i, k] + phi * (population[i, k] - population[j, k])
            
            # Đảm bảo giá trị mới không vượt ra ngoài biên
            candidate_solution[k] = np.clip(candidate_solution[k], lb[k], ub[k])
            
            # Đánh giá lời giải ứng viên
            candidate_fitness = fitness_function(candidate_solution)
            
            # Lựa chọn tham lam
            if candidate_fitness > fitness_values[i]:
                population[i] = candidate_solution
                fitness_values[i] = candidate_fitness
                trials_counter[i] = 0
            else:
                trials_counter[i] += 1

        # --- GIAI ĐOẠN 2: ONG QUAN SÁT (ONLOOKER BEES) ---
        # Áp dụng Công thức 2: P_i = fit_i / sum(fit_j)
        # Để tránh xác suất âm, ta dịch chuyển fitness values về không âm
        min_fitness = np.min(fitness_values)
        if min_fitness < 0:
            shifted_fitness = fitness_values - min_fitness  # Dịch về >= 0
        else:
            shifted_fitness = fitness_values
        
        total_fitness = np.sum(shifted_fitness)
        if total_fitness == 0:
            probabilities = np.ones(n_solutions) / n_solutions # Chia đều xác suất
        else:
            probabilities = shifted_fitness / total_fitness
            
        # Vòng lặp của các Ong Quan sát
        for _ in range(n_solutions): # Có SN con ong quan sát
            # Chọn một lời giải dựa trên xác suất (vòng quay roulette)
            selected_index = np.random.choice(range(n_solutions), p=probabilities)
            
            # ----- HÀNH ĐỘNG CỦA ONG QUAN SÁT -----
            # (Lặp lại chính xác logic của Ong Thợ cho lời giải đã chọn)
            
            # Chọn "hàng xóm" ngẫu nhiên j khác selected_index
            j_list = list(range(n_solutions))
            j_list.remove(selected_index)
            j = np.random.choice(j_list)
            k = np.random.randint(0, n_dimensions)
            phi = np.random.uniform(-1, 1)
            
            candidate_solution = population[selected_index].copy()
            # Áp dụng lại Công thức 1
            candidate_solution[k] = population[selected_index, k] + phi * (population[selected_index, k] - population[j, k])
            candidate_solution[k] = np.clip(candidate_solution[k], lb[k], ub[k])
            
            candidate_fitness = fitness_function(candidate_solution)
            
            if candidate_fitness > fitness_values[selected_index]:
                population[selected_index] = candidate_solution
                fitness_values[selected_index] = candidate_fitness
                trials_counter[selected_index] = 0
            else:
                trials_counter[selected_index] += 1
                
        # --- GIAI ĐOẠN 3: ONG TRINH SÁT (SCOUT BEES) ---
        for i in range(n_solutions):
            if trials_counter[i] > limit:
                # Thay thế lời giải bị bế tắc bằng một lời giải ngẫu nhiên mới
                # Áp dụng Công thức 3
                population[i] = lb + np.random.rand(n_dimensions) * (ub - lb)
                fitness_values[i] = fitness_function(population[i])
                trials_counter[i] = 0
        
        # --- Ghi nhớ lời giải tốt nhất cho đến nay ---
        current_best_fitness = np.max(fitness_values)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[np.argmax(fitness_values)].copy()
        
        if (iteration + 1) % 10 == 0:
            print(f"Loop {iteration + 1} | Best fitness: {best_fitness:.4f}")
            
    return best_solution, best_fitness

# --- CÁCH CHẠY VÀ KẾT QUẢ ---
if __name__ == "__main__":
    # 1. Định nghĩa hàm fitness cho bài toán
    # 2. Thiết lập các tham số
    bounds = [(-20, 20), (-20, 20)]  # Biên cho x và y
    SN = 50                         # Số lượng lời giải (quy mô đàn)
    N_DIM = 2                       # Số chiều (x và y)
    LIMIT = 100                     # Ngưỡng từ bỏ
    MAX_ITER = 200                  # Số vòng lặp tối đa

    # 3. Chạy thuật toán ABC
    print("Start...")
    solution, fitness = run_abc(objective_function, bounds, SN, N_DIM, LIMIT, MAX_ITER)

    # 4. In kết quả
    print("\n--- END ---")
    print(f"Best solution found at:")
    print(f"  x = {solution[0]:.4f}")
    print(f"  y = {solution[1]:.4f}")
    print(f"With maximum fitness value: {fitness:.4f}")
    print("(Theoretical answer is x=10, y=-5, fitness=100)")