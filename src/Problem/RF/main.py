"""
Demo script for visualizing the Rosenbrock function.
"""
from RF import RF


if __name__ == "__main__":
    # Create Rosenbrock function with a=1, b=100
    rf = RF(a=1, b=100)

    # Define sample points to visualize
    nodes = [
        (-1, 1), 
        (0, 0), 
        (0.5, 0.2),
    ]

    # Visualize in 2D contour plot
    print("Generating 2D contour plot...")
    rf.visualize(points=nodes, three_d=False)
    
    # Visualize in 3D surface plot
    print("Generating 3D surface plot...")
    rf.visualize(points=nodes, three_d=True)
