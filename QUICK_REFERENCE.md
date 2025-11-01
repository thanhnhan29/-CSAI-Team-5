# Quick Reference Guide

## 🚀 Getting Started (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Try Examples
```bash
python example.py
```

### Step 3: Run Full Tests
```bash
# Test on Rosenbrock Function
python tests/test_rosenbrock.py

# Test on Knapsack Problem
python tests/test_knapsack.py

# Run ALL experiments (both problems)
python experiments/run_experiments.py
```

---

## 📚 Algorithm Parameters

### Firefly Algorithm
```python
from algorithms.firefly import FireflyAlgorithm

fa = FireflyAlgorithm(
    n_fireflies=30,    # Population size (20-50 recommended)
    max_iter=500,      # Maximum iterations (100-1000)
    alpha=0.5,         # Randomization [0, 1] (0.2-0.8)
    beta0=1.0,         # Attractiveness [0, 2] (0.5-2.0)
    gamma=1.0,         # Absorption [0.01, 100] (0.1-10)
    mode='continuous', # 'continuous' or 'binary'
    seed=42            # Random seed (optional)
)
```

### Cuckoo Search
```python
from algorithms.cuckoo_search import CuckooSearch

cs = CuckooSearch(
    n_nests=30,        # Population size (20-50 recommended)
    max_iter=500,      # Maximum iterations (100-1000)
    pa=0.25,           # Discovery rate [0, 1] (0.1-0.5)
    levy_lambda=1.5,   # Lévy exponent [1, 3] (1.5 recommended)
    mode='continuous', # 'continuous' or 'binary'
    seed=42            # Random seed (optional)
)
```

---

## 🎯 Problem Examples

### Rosenbrock Function (Continuous)
```python
from problems.rosenbrock import RosenbrockFunction

problem = RosenbrockFunction(
    n_dim=10,          # Dimension (2-50)
    bounds=(-5, 5)     # Search space
)

# Evaluate
fitness = problem.evaluate([1, 1, 1, 1, 1])  # Returns 0.0

# Get bounds
lower, upper = problem.get_bounds()
```

### Knapsack Problem (Discrete)
```python
from problems.knapsack import KnapsackProblem

# Option 1: Generate random instance
problem = KnapsackProblem(n_items=50, seed=42)

# Option 2: Use custom data
weights = [10, 20, 30]
values = [60, 100, 120]
capacity = 50
problem = KnapsackProblem(weights=weights, values=values, capacity=capacity)

# Evaluate binary solution
solution = [1, 0, 1, 0, 1, ...]  # Binary vector
fitness = problem.evaluate(solution)

# Get solution info
info = problem.get_solution_info(solution)
print(info['total_value'])
print(info['is_feasible'])
```

---

## 🔬 Running Optimization

### Basic Template
```python
from algorithms.firefly import FireflyAlgorithm
from problems.rosenbrock import RosenbrockFunction

# 1. Create problem
problem = RosenbrockFunction(n_dim=10)

# 2. Create algorithm
algorithm = FireflyAlgorithm(n_fireflies=30, max_iter=500)

# 3. Optimize
best_solution, best_fitness, history = algorithm.optimize(
    objective_func=problem.evaluate,
    n_dim=problem.n_dim,
    bounds=problem.get_bounds()
)

# 4. Display results
print(f"Best fitness: {best_fitness}")
print(f"Best solution: {best_solution}")
print(f"Convergence history: {history['best_fitness']}")
```

---

## 📊 Visualization

### Plot Convergence
```python
from utils.visualization import plot_convergence

plot_convergence(
    histories=[fa_history, cs_history],
    labels=['Firefly Algorithm', 'Cuckoo Search'],
    title='Convergence Comparison',
    save_path='results/my_plot.png',
    show=True
)
```

### Compare Algorithms
```python
from utils.visualization import plot_comparison

results = {
    'FA': [100, 120, 95, 110, 105],  # Multiple run results
    'CS': [85, 90, 80, 88, 92]
}

plot_comparison(
    results=results,
    metric='distribution',  # or 'best'
    title='Algorithm Comparison',
    save_path='results/comparison.png'
)
```

---

## 🎨 Visualization Outputs

After running experiments, check `results/` directory:

### Rosenbrock Function
- `rosenbrock_surface.png` - 3D visualization
- `rosenbrock_convergence_best.png` - Best runs
- `rosenbrock_convergence_all.png` - All runs with confidence
- `rosenbrock_comparison_box.png` - Box plot
- `rosenbrock_comparison_bar.png` - Bar chart
- `rosenbrock_statistics.png` - Statistics table

### Knapsack Problem
- `knapsack_convergence_best.png` - Best runs
- `knapsack_convergence_all.png` - All runs with confidence
- `knapsack_comparison_box.png` - Box plot
- `knapsack_comparison_bar.png` - Bar chart
- `knapsack_statistics.png` - Statistics table

---

## 🔧 Customization Tips

### Improve Performance
```python
# For better exploration:
alpha = 0.8          # Higher randomization
n_fireflies = 50     # Larger population

# For better exploitation:
alpha = 0.2          # Lower randomization
max_iter = 1000      # More iterations
```

### Solve Custom Problem
```python
import numpy as np

# 1. Define objective function
def my_function(x):
    # Your optimization objective
    return np.sum(x**2) + 10

# 2. Set parameters
n_dim = 5
bounds = (np.full(n_dim, -10), np.full(n_dim, 10))

# 3. Optimize
from algorithms.firefly import FireflyAlgorithm
fa = FireflyAlgorithm(n_fireflies=30, max_iter=200)
best_sol, best_fit, history = fa.optimize(my_function, n_dim, bounds)

print(f"Best solution: {best_sol}")
print(f"Best fitness: {best_fit}")
```

---

## 📈 Performance Tuning

### Rosenbrock Function
**Recommended Settings:**
- Population: 30-50
- Iterations: 500-1000
- Alpha (FA): 0.3-0.5
- Pa (CS): 0.2-0.3

### Knapsack Problem
**Recommended Settings:**
- Population: 30-50
- Iterations: 500-1000
- Mode: 'binary'
- Alpha (FA): 0.5-0.7
- Pa (CS): 0.2-0.3

---

## 🐛 Troubleshooting

### Import Errors
```bash
# Make sure you're in the project directory
cd -CSAI-Team-5

# Run from project root
python example.py
```

### Missing Packages
```bash
pip install --upgrade numpy matplotlib seaborn
```

### Slow Performance
- Reduce population size
- Reduce max iterations
- Use fewer runs in tests

---

## 📞 Support

- Check `README.md` for detailed documentation
- See `IMPLEMENTATION_SUMMARY.md` for technical details
- Review `example.py` for usage examples

---

## ⚡ Quick Commands Cheat Sheet

```bash
# Installation
pip install -r requirements.txt

# Examples
python example.py

# Individual Tests
python tests/test_rosenbrock.py
python tests/test_knapsack.py

# Full Experiments
python experiments/run_experiments.py

# Check Results
ls results/
```

---

**Last Updated:** November 1, 2025
