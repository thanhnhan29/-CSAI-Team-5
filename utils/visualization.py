"""
Visualization utilities for optimization results.

Uses Matplotlib and Seaborn for creating plots and graphs.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_convergence(histories, labels, title="Convergence Curve", 
                     save_path=None, show=True):
    """
    Plot convergence curves for multiple algorithms.
    
    Parameters:
    -----------
    histories : list of dict
        List of history dictionaries from optimization runs
    labels : list of str
        Labels for each algorithm
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    show : bool
        Whether to display the plot
    """
    plt.figure(figsize=(12, 6))
    
    for history, label in zip(histories, labels):
        iterations = range(len(history['best_fitness']))
        plt.plot(iterations, history['best_fitness'], label=label, linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Fitness', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison(results, metric='best', title="Algorithm Comparison",
                   save_path=None, show=True):
    """
    Create box plot or bar chart comparing algorithms.
    
    Parameters:
    -----------
    results : dict
        Dictionary with algorithm names as keys and list of fitness values
        Example: {'FA': [fit1, fit2, ...], 'CS': [fit1, fit2, ...]}
    metric : str
        'best' for bar chart of best values, 'distribution' for box plot
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    show : bool
        Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if metric == 'distribution':
        # Box plot for distribution
        data = [values for values in results.values()]
        labels = list(results.keys())
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        notch=True, showmeans=True)
        
        # Color the boxes
        colors = sns.color_palette("Set2", len(labels))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Fitness Value', fontsize=12)
        ax.set_xlabel('Algorithm', fontsize=12)
        
    else:  # metric == 'best'
        # Bar chart for best values
        algorithms = list(results.keys())
        best_values = [np.min(values) for values in results.values()]
        mean_values = [np.mean(values) for values in results.values()]
        std_values = [np.std(values) for values in results.values()]
        
        x = np.arange(len(algorithms))
        width = 0.35
        
        colors = sns.color_palette("Set2", len(algorithms))
        bars1 = ax.bar(x - width/2, best_values, width, label='Best',
                       color=colors, alpha=0.8)
        bars2 = ax.bar(x + width/2, mean_values, width, label='Mean',
                       color=colors, alpha=0.5)
        
        # Add error bars for mean
        ax.errorbar(x + width/2, mean_values, yerr=std_values,
                   fmt='none', color='black', capsize=5, alpha=0.5)
        
        ax.set_ylabel('Fitness Value', fontsize=12)
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms)
        ax.legend(fontsize=10)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_rosenbrock_surface(n_points=100, bounds=(-2, 2), save_path=None, show=True):
    """
    Plot 3D surface of Rosenbrock function (2D case).
    
    Parameters:
    -----------
    n_points : int
        Number of points per dimension
    bounds : tuple
        (min, max) for both dimensions
    save_path : str, optional
        Path to save the figure
    show : bool
        Whether to display the plot
    """
    from problems.rosenbrock import RosenbrockFunction
    
    # Create meshgrid
    x1 = np.linspace(bounds[0], bounds[1], n_points)
    x2 = np.linspace(bounds[0], bounds[1], n_points)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Evaluate function
    rosenbrock = RosenbrockFunction(n_dim=2)
    Z = np.zeros_like(X1)
    for i in range(n_points):
        for j in range(n_points):
            Z[i, j] = rosenbrock.evaluate([X1[i, j], X2[i, j]])
    
    # Use log scale for better visualization
    Z_log = np.log10(Z + 1)
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 6))
    
    # 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X1, X2, Z_log, cmap='viridis', alpha=0.8,
                           edgecolor='none')
    ax1.set_xlabel('$x_1$', fontsize=12)
    ax1.set_ylabel('$x_2$', fontsize=12)
    ax1.set_zlabel('$log_{10}(f(x) + 1)$', fontsize=12)
    ax1.set_title('Rosenbrock Function - 3D Surface', fontsize=12, fontweight='bold')
    fig.colorbar(surf, ax=ax1, shrink=0.5)
    
    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X1, X2, Z_log, levels=20, cmap='viridis')
    ax2.contour(X1, X2, Z_log, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    ax2.plot(1, 1, 'r*', markersize=15, label='Global Optimum')
    ax2.set_xlabel('$x_1$', fontsize=12)
    ax2.set_ylabel('$x_2$', fontsize=12)
    ax2.set_title('Rosenbrock Function - Contour Plot', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    fig.colorbar(contour, ax=ax2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_multiple_runs(all_histories, algorithm_names, title="Multiple Runs",
                       save_path=None, show=True):
    """
    Plot convergence curves for multiple runs of algorithms.
    
    Parameters:
    -----------
    all_histories : dict
        Dictionary with algorithm names as keys and lists of history dicts
        Example: {'FA': [hist1, hist2, ...], 'CS': [hist1, hist2, ...]}
    algorithm_names : list of str
        Names of algorithms to plot
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    show : bool
        Whether to display the plot
    """
    plt.figure(figsize=(12, 6))
    
    colors = sns.color_palette("Set2", len(algorithm_names))
    
    for algo_name, color in zip(algorithm_names, colors):
        histories = all_histories[algo_name]
        
        # Calculate statistics across runs
        max_len = max(len(h['best_fitness']) for h in histories)
        all_fitness = []
        
        for history in histories:
            fitness = history['best_fitness']
            # Pad if necessary
            if len(fitness) < max_len:
                fitness = fitness + [fitness[-1]] * (max_len - len(fitness))
            all_fitness.append(fitness)
        
        all_fitness = np.array(all_fitness)
        mean_fitness = np.mean(all_fitness, axis=0)
        std_fitness = np.std(all_fitness, axis=0)
        
        iterations = range(len(mean_fitness))
        
        # Plot mean with confidence interval
        plt.plot(iterations, mean_fitness, label=algo_name, color=color, linewidth=2)
        plt.fill_between(iterations, 
                        mean_fitness - std_fitness,
                        mean_fitness + std_fitness,
                        alpha=0.2, color=color)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Fitness', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_statistics_table(results_dict, save_path=None, show=True):
    """
    Create a table with statistics for each algorithm.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with algorithm names and their results
    save_path : str, optional
        Path to save the figure
    show : bool
        Whether to display the plot
    """
    # Calculate statistics
    stats = {}
    for algo_name, fitness_values in results_dict.items():
        stats[algo_name] = {
            'Best': np.min(fitness_values),
            'Worst': np.max(fitness_values),
            'Mean': np.mean(fitness_values),
            'Median': np.median(fitness_values),
            'Std': np.std(fitness_values)
        }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    columns = ['Algorithm', 'Best', 'Worst', 'Mean', 'Median', 'Std']
    table_data = []
    for algo_name, stat in stats.items():
        row = [algo_name,
               f"{stat['Best']:.6f}",
               f"{stat['Worst']:.6f}",
               f"{stat['Mean']:.6f}",
               f"{stat['Median']:.6f}",
               f"{stat['Std']:.6f}"]
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=columns,
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.17, 0.17, 0.17, 0.17, 0.17])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        if i % 2 == 0:
            for j in range(len(columns)):
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Statistical Results', fontsize=14, fontweight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
