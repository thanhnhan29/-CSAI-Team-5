# Project 1 - Team 5

<<<<<<< HEAD
## Swarm Intelligence Algorithms Implementation

This project implements and compares two swarm intelligence algorithms:
- **Firefly Algorithm (FA)**
- **Cuckoo Search (CS)**

on two optimization problems:
- **Continuous Optimization:** Rosenbrock Function
- **Discrete Optimization:** 0/1 Knapsack Problem

---

## Project Structure

```
-CSAI-Team-5/
├── algorithms/           # Algorithm implementations
│   ├── __init__.py
│   ├── firefly.py       # Firefly Algorithm (continuous & binary)
│   └── cuckoo_search.py # Cuckoo Search (continuous & binary)
├── problems/            # Problem definitions
│   ├── __init__.py
│   ├── rosenbrock.py    # Rosenbrock function
│   └── knapsack.py      # Knapsack problem
├── utils/               # Utility functions
│   ├── __init__.py
│   └── visualization.py # Plotting and visualization
├── tests/               # Individual test scripts
│   ├── __init__.py
│   ├── test_rosenbrock.py
│   └── test_knapsack.py
├── experiments/         # Main experiment runner
│   └── run_experiments.py
├── results/             # Output directory (auto-created)
├── requirements.txt     # Dependencies
└── README.md           # This file
```

---

## Technical Requirements

✅ **NumPy only:** All algorithms implemented from scratch using only NumPy (no scikit-learn or scipy.optimize)

✅ **Configurable parameters:** Each algorithm has customizable:
   - Population size (n_fireflies, n_nests)
   - Maximum iterations
   - Algorithm-specific parameters (alpha, beta0, gamma, pa, levy_lambda)
   - Mode (continuous/binary)

✅ **Visualization:** Uses Matplotlib and Seaborn for plots

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd -CSAI-Team-5
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Required packages:
   - numpy >= 1.21.0
   - matplotlib >= 3.4.0
   - seaborn >= 0.11.0

---

## Usage

### Quick Start - Run All Experiments

To run comprehensive experiments on both problems:

```bash
python experiments/run_experiments.py
```

This will:
- Run 30 independent tests for each algorithm on each problem
- Generate statistical summaries
- Create visualizations (saved to `results/` directory)
- Display comprehensive results

**Expected runtime:** ~5-10 minutes (depending on your machine)

### Individual Tests

#### Test on Rosenbrock Function:
```bash
python tests/test_rosenbrock.py
```

Configuration:
- Dimension: 10D
- Runs: 30
- Iterations: 500
- Population: 30

#### Test on Knapsack Problem:
```bash
python tests/test_knapsack.py
```

Configuration:
- Items: 50
- Runs: 30
- Iterations: 500
- Population: 30

---

## Algorithm Details

### Firefly Algorithm (FA)

**Inspiration:** Flashing patterns of fireflies

**Key Parameters:**
- `n_fireflies`: Population size (default: 20)
- `max_iter`: Maximum iterations (default: 100)
- `alpha`: Randomization parameter (default: 0.5)
- `beta0`: Attractiveness at r=0 (default: 1.0)
- `gamma`: Light absorption coefficient (default: 1.0)
- `mode`: 'continuous' or 'binary'

**Key Concepts:**
- Attractiveness: β(r) = β₀ × e^(-γr²)
- Fireflies move toward brighter ones
- Binary mode uses sigmoid transfer function

### Cuckoo Search (CS)

**Inspiration:** Brood parasitism of cuckoo birds

**Key Parameters:**
- `n_nests`: Population size (default: 25)
- `max_iter`: Maximum iterations (default: 100)
- `pa`: Discovery probability (default: 0.25)
- `levy_lambda`: Lévy flight parameter (default: 1.5)
- `mode`: 'continuous' or 'binary'

**Key Concepts:**
- Lévy flights for global exploration
- Fraction of worst nests abandoned
- Binary mode uses sigmoid transfer function

---

## Problem Details

### Rosenbrock Function

**Formula:** f(x) = Σ[100(x_{i+1} - x_i²)² + (1 - x_i)²]

**Properties:**
- Dimension: Configurable (default: 10D)
- Search space: [-5, 5]ⁿ
- Global minimum: f(1, 1, ..., 1) = 0
- Challenge: Narrow valley to optimum

### Knapsack Problem (0/1)

**Objective:** Maximize total value subject to weight constraint

**Properties:**
- Binary decision variables (include/exclude items)
- Capacity constraint
- NP-hard problem
- Penalty for constraint violations

---

## Results and Visualizations

After running experiments, the following plots are generated in `results/`:

### Rosenbrock Function:
1. `rosenbrock_surface.png` - 3D surface and contour plot
2. `rosenbrock_convergence_best.png` - Best run convergence
3. `rosenbrock_convergence_all.png` - All runs with confidence intervals
4. `rosenbrock_boxplot.png` - Distribution comparison
5. `rosenbrock_statistics.png` - Statistical summary table

### Knapsack Problem:
1. `knapsack_convergence_best.png` - Best run convergence
2. `knapsack_convergence_all.png` - All runs with confidence intervals
3. `knapsack_boxplot.png` - Value distribution comparison
4. `knapsack_statistics.png` - Statistical summary table

---

## Example Usage in Python

### Using Firefly Algorithm on Custom Problem:

```python
from algorithms.firefly import FireflyAlgorithm
import numpy as np

# Define objective function
def sphere_function(x):
    return np.sum(x**2)

# Initialize algorithm
fa = FireflyAlgorithm(
    n_fireflies=30,
    max_iter=200,
    alpha=0.5,
    beta0=1.0,
    gamma=1.0,
    mode='continuous'
)

# Run optimization
n_dim = 5
bounds = (np.full(n_dim, -10), np.full(n_dim, 10))
best_solution, best_fitness, history = fa.optimize(
    objective_func=sphere_function,
    n_dim=n_dim,
    bounds=bounds
)

print(f"Best fitness: {best_fitness}")
print(f"Best solution: {best_solution}")
```

### Using Cuckoo Search on Knapsack:

```python
from algorithms.cuckoo_search import CuckooSearch
from problems.knapsack import KnapsackProblem

# Create problem instance
problem = KnapsackProblem(n_items=30, seed=42)

# Initialize algorithm
cs = CuckooSearch(
    n_nests=25,
    max_iter=300,
    pa=0.25,
    levy_lambda=1.5,
    mode='binary'
)

# Run optimization
best_solution, best_fitness, history = cs.optimize(
    objective_func=problem.evaluate,
    n_dim=problem.n_items,
    bounds=problem.get_bounds()
)

# Get solution details
info = problem.get_solution_info(best_solution)
print(f"Total value: {info['total_value']}")
print(f"Total weight: {info['total_weight']}")
print(f"Feasible: {info['is_feasible']}")
```

---

## Performance Metrics

Each test run reports:
- **Best fitness:** Best value found
- **Mean fitness:** Average over all runs
- **Median fitness:** Median value
- **Std:** Standard deviation
- **Convergence speed:** Iterations to reach near-optimal
- **Computation time:** Average time per run

---

## Customization

### Adjusting Algorithm Parameters:

Edit the configuration in test scripts or experiment runner:

```python
# Firefly Algorithm
fa = FireflyAlgorithm(
    n_fireflies=50,      # Increase population
    max_iter=1000,       # More iterations
    alpha=0.2,           # Less randomization
    beta0=2.0,           # Higher attractiveness
    gamma=0.5            # Less absorption
)

# Cuckoo Search
cs = CuckooSearch(
    n_nests=40,          # Increase population
    max_iter=1000,       # More iterations
    pa=0.1,              # Less abandonment
    levy_lambda=2.0      # Different Lévy exponent
)
```

### Creating Custom Problems:

Implement a class with these methods:
- `evaluate(solution)` - Returns fitness value
- `get_bounds()` - Returns (lower, upper) bounds
- `n_dim` - Problem dimension

---

## Contributors

Team 5

---

## License

[Add your license information here]

---

## References

1. Yang, X. S. (2009). Firefly algorithms for multimodal optimization. In International symposium on stochastic algorithms (pp. 169-178). Springer, Berlin, Heidelberg.

2. Yang, X. S., & Deb, S. (2009). Cuckoo search via Lévy flights. In 2009 World congress on nature & biologically inspired computing (NaBIC) (pp. 210-214). IEEE.
=======
## Mô tả Project

Project này bao gồm các thuật toán tối ưu hóa AI:

- **ACO (Ant Colony Optimization)**: Thuật toán tối ưu bầy đàn kiến
- **Simulated Annealing**: Thuật toán luyện kim mô phỏng
- **Knapsack Problem (KP)**: Bài toán cái túi
- **Rosenbrock Function (RF)**: Hàm tối ưu Rosenbrock

## Yêu cầu hệ thống

- Python >= 3.7
- pip (Python package manager)

## Hướng dẫn Setup Project

### Bước 1: Clone repository (nếu chưa có)

```bash
git clone <repository-url>
cd CSAI
```

### Bước 2: Tạo môi trường ảo (khuyến nghị)

```bash
# Tạo virtual environment
python3 -m venv venv

# Kích hoạt virtual environment
# Trên macOS/Linux:
source venv/bin/activate

# Trên Windows:
# venv\Scripts\activate
```

### Bước 3: Cài đặt dependencies từ requirements.txt

```bash
pip install -r requirements.txt
```

File `requirements.txt` bao gồm các thư viện:

- numpy: Tính toán số học
- matplotlib: Vẽ đồ thị
- seaborn: Visualization nâng cao
- tqdm: Progress bar

### Bước 4: Cài đặt package dưới dạng editable mode

```bash
pip install -e .
```

Lệnh này sẽ:

- Đọc file `pyproject.toml`
- Cài đặt package `csai` ở chế độ development
- Cho phép bạn import các module từ bất kỳ đâu trong project

### Bước 5: Chạy chương trình

```bash
# Di chuyển vào thư mục src
cd src

# Chạy file main
python main.py
```

## Cấu trúc Project

```
CSAI/
├── README.md                 # File hướng dẫn này
├── requirements.txt          # Danh sách thư viện cần thiết
├── pyproject.toml           # Cấu hình build package
├── doc/                     # Tài liệu
├── src/                     # Source code chính
│   ├── main.py             # File chạy chính
│   ├── aco/                # Ant Colony Optimization
│   │   ├── __init__.py
│   │   ├── ant.py
│   │   └── colony.py
│   ├── simulated_annealing/ # Simulated Annealing
│   │   ├── _init__.py
│   │   └── simanneal.py
│   ├── Problem/            # Các bài toán tối ưu
│   │   ├── KP/            # Knapsack Problem
│   │   └── RF/            # Rosenbrock Function
│   └── Test/              # Test files
└── venv/                   # Virtual environment (sau khi tạo)
```

## Chạy các module riêng lẻ

### Test Rosenbrock Function

```bash
cd src/Test
python RF_test.py
```

### Chạy ACO cho Knapsack Problem

```bash
cd src
python main.py
```

## Troubleshooting

### Lỗi: Module not found

Nếu gặp lỗi `ModuleNotFoundError`, hãy đảm bảo:

1. Đã cài đặt tất cả dependencies: `pip install -r requirements.txt`
2. Đã cài package ở editable mode: `pip install -e .`
3. Virtual environment đã được kích hoạt

### Lỗi: Permission denied

Trên macOS/Linux, có thể cần dùng `python3` thay vì `python`:

```bash
python3 -m pip install -r requirements.txt
```

## Ghi chú

- Luôn kích hoạt virtual environment trước khi làm việc với project
- Sau khi thay đổi code, không cần cài lại vì đã dùng editable mode (`-e`)
- Để thoát virtual environment: `deactivate`

## Team Members

Team 5 - CSAI Project

## License

[Thêm license nếu có]
>>>>>>> merge-test
