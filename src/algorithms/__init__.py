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

from .swarm_intelligence.aco import Ant, ACO
from .swarm_intelligence.abc import run_abc
from .swarm_intelligence.pso import Particle, PSO
from .traditional_search.simulated_annealing import SimulatedAnnealing
from .traditional_search.hill_climbing import steepest_ascent_hill_climbing
from .traditional_search.search import Node, GraphSearch

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