import numpy as np
import matplotlib.pyplot as plt

class RF:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, x, y):
        return (self.a - x) ** 2 + self.b * (y - x**2) ** 2
    
    def visualize(self, xlim=(-2, 2), ylim=(-1, 3), resolution=400, three_d=False, points=None):
        """
        Visualize the Rosenbrock function, optionally with given points and global minimum.
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
        """Return the global minimum (x*, y*, f(x*, y*))."""
        x_min = self.a
        y_min = self.a ** 2
        f_min = self.__call__(x_min, y_min)
        return (x_min, y_min, f_min)