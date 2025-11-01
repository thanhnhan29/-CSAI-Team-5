"""
CSAI Algorithms Package
=======================

This package contains implementations of various optimization and search algorithms:
- ACO (Ant Colony Optimization)
- ABC (Artificial Bee Colony)
- PSO (Particle Swarm Optimization)
- Simulated Annealing
- Hill Climbing
- Graph Search (BFS, DFS, A*)
"""

from .aco import Ant, ACO
from .abc import run_abc
from .pso import Particle, PSO
from .simulated_annealing import SimulatedAnnealing
from .hill_climbing import steepest_ascent_hill_climbing
from .search import Node, GraphSearch

__all__ = [
    'Ant',
    'ACO',
    'run_abc',
    'Particle',
    'PSO',
    'SimulatedAnnealing',
    'steepest_ascent_hill_climbing',
    'Node',
    'GraphSearch',
]

__version__ = '0.1.0' 