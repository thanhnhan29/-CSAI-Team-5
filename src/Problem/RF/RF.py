import numpy as np
import matplotlib.pyplot as plt

class RF:
    def __init__(self, a=1, b=1.5):
        self.a = a
        self.b = b

    def __call__(self, x, y):
        x = np.array(x)
        y = np.array(y)
        result = (self.a - x) ** 2 + self.b * (y - x**2) ** 2
        return float(result) if result.size == 1 else result
    
    def visualize(self, xlim=None, ylim=None, resolution=400, three_d=False, points=None, gradient_color=False):

        x_min, y_min, f_min = self.global_minimum()

        if xlim is None:
            if points is not None and len(points) > 0:
                points = np.array(points, dtype=float)
                xlim = (min(x_min - 1, np.min(points[:, 0]) - 1),
                        max(x_min + 1, np.max(points[:, 0]) + 1))
                ylim = (min(y_min - 1, np.min(points[:, 1]) - 1),
                        max(y_min + 1, np.max(points[:, 1]) + 1))
            else:
                xlim = (x_min - 1, x_min + 1)
                ylim = (y_min - 1, y_min + 1)

        x = np.linspace(*xlim, resolution)
        y = np.linspace(*ylim, resolution)
        X, Y = np.meshgrid(x, y)
        Z = self.__call__(X, Y)
        if three_d:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
            ax.set_title(f"Rosenbrock Function (a={self.a}, b={self.b})")
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('f(x, y)')

            ax.scatter(x_min, y_min, f_min, color='red', s=70, label='Global Minimum', depthshade=True)

            if points is not None:
                pts = np.array(points)
                Z_pts = self.__call__(pts[:, 0], pts[:, 1])

                if gradient_color and len(points) > 1:
                    colors = plt.cm.plasma(np.linspace(0, 1, len(points)))
                    for i, c in enumerate(colors):
                        ax.scatter(pts[i, 0], pts[i, 1], Z_pts[i], color=c,cmap='viridis', s=50)
                else:
                    ax.scatter(pts[:, 0], pts[:, 1], Z_pts, color='red', s=50, label='Nodes')

            ax.legend()

        else:
            fig, ax = plt.subplots(figsize=(7, 6))
            levels = np.logspace(-0.5, 3, 20)
            contour = ax.contour(X, Y, Z, levels=levels, cmap='viridis')
            ax.set_title(f"Rosenbrock Function Contour (a={self.a}, b={self.b})")
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            fig.colorbar(contour, ax=ax, label='f(x, y)')


            if points is not None and len(points) > 0:
                pts = np.array(points)
                if gradient_color and len(points) > 1:
                    scatter = ax.scatter(
                        pts[:, 0], pts[:, 1],
                        c=np.linspace(0, 255, len(points)),
                        cmap='viridis',
                        s=40,
                        label='Nodes'
                    )
                    cbar = fig.colorbar(scatter, ax=ax, label='Iteration Progress')
                    cbar.set_ticks([0, 255])
                    cbar.set_ticklabels(['Start', 'End'])
                else:
                    ax.scatter(pts[:, 0], pts[:, 1], color='blue', s=40, label='Nodes')

            ax.scatter(x_min, y_min, color='red', s=60, label='Global Minimum')
            ax.legend()

        plt.show()

    def global_minimum(self):
        """Return the global minimum (x*, y*, f(x*, y*))."""
        x_min = self.a
        y_min = self.a ** 2
        f_min = self.__call__(x_min, y_min)
        return (x_min, y_min, f_min)
    
    def loss(self, y_pred):
        _, _, y = self.global_minimum()
        result = (y_pred - y)**2
        
        if isinstance(y_pred, (float, int)):
            return float(result)
        else:
            return [abs(v - y) for v in y_pred]