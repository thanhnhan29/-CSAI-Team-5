"""
Rosenbrock Function Implementation.

The Rosenbrock function is a non-convex function used as a performance test 
problem for optimization algorithms.
"""
import numpy as np
import matplotlib.pyplot as plt


class RF:
    """
    Rosenbrock Function class.
    
    The Rosenbrock function is defined as:
    f(x, y) = (a - x)^2 + b(y - x^2)^2
    
    The global minimum is at (a, a^2) with f(a, a^2) = 0.
    """
    
    def __init__(self, a, b):
        """
        Initialize the Rosenbrock function.
        
        Args:
            a: Parameter a in the Rosenbrock function.
            b: Parameter b in the Rosenbrock function.
        """
        self.a = a
        self.b = b

    def __call__(self, x, y):
        """
        Evaluate the Rosenbrock function at (x, y).
        
        Args:
            x: x-coordinate(s).
            y: y-coordinate(s).
            
        Returns:
            Function value(s) at (x, y).
        """
        return (self.a - x) ** 2 + self.b * (y - x**2) ** 2
    
    def visualize(self, xlim=(-2, 2), ylim=(-1, 3), resolution=400, three_d=False, points=None):
        """
        Visualize the Rosenbrock function, optionally with given points and global minimum.
        
        Args:
            xlim: x-axis limits, tuple of (min, max).
            ylim: y-axis limits, tuple of (min, max).
            resolution: Grid resolution for plotting.
            three_d: If True, create 3D surface plot. If False, create 2D contour plot.
            points: Optional list of (x, y) points to plot on the function surface.
        """
        x = np.linspace(*xlim, resolution)
        y = np.linspace(*ylim, resolution)
        X, Y = np.meshgrid(x, y)
        Z = self.__call__(X, Y)

        x_min, y_min, f_min = self.global_minimum()

        if three_d:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
            ax.set_title(f"Rosenbrock Function (a={self.a}, b={self.b})")
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('f(x, y)')

            ax.scatter(x_min, y_min, f_min, color='blue', s=70, label='Global Minimum', depthshade=True)

            if points is not None:
                pts = np.array(points)
                Z_pts = self.__call__(pts[:, 0], pts[:, 1])
                ax.scatter(pts[:, 0], pts[:, 1], Z_pts, color='red', s=50, label='Nodes')

            ax.legend()

        else:
            plt.figure(figsize=(7, 6))
            levels = np.logspace(-0.5, 3, 20)
            plt.contour(X, Y, Z, levels=levels, cmap='viridis')
            plt.title(f"Rosenbrock Function Contour (a={self.a}, b={self.b})")
            plt.xlabel('x')
            plt.ylabel('y')
            plt.colorbar(label='f(x, y)')

            plt.scatter(x_min, y_min, color='blue', s=60, label='Global Minimum')

            if points is not None:
                pts = np.array(points)
                plt.scatter(pts[:, 0], pts[:, 1], color='red', s=40, label='Nodes')

            plt.legend()

        plt.show()

    def global_minimum(self):
        """
        Return the global minimum of the Rosenbrock function.
        
        Returns:
            tuple: (x*, y*, f(x*, y*)) where (x*, y*) is the global minimum.
        """
        x_min = self.a
        y_min = self.a ** 2
        f_min = self.__call__(x_min, y_min)
        return (x_min, y_min, f_min)
