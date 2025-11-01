import numpy as np
from .particle import Particle

class PSO:
    def __init__(self, num_particles, num_iter, num_dimensions, bounds,
                 compute_fitness, particle_type='continuous',
                 w=0.7, c1=1.5, c2=1.5, w_decay=0.99, 
                 repair_func=None, **particle_kwargs):
        """
        Initialize PSO algorithm
        
        Args:
            num_particles: Number of particles in the swarm
            num_iter: Number of iterations
            num_dimensions: Number of dimensions in search space
            bounds: List of tuples [(min, max), ...] for each dimension
            compute_fitness: Fitness function (higher is better)
            particle_type: 'continuous' or 'binary'
            w: Inertia weight (typical: 0.7-0.9)
            c1: Cognitive parameter - attraction to personal best (typical: 1.5-2.0)
            c2: Social parameter - attraction to global best (typical: 1.5-2.0)
            w_decay: Inertia weight decay per iteration (typical: 0.95-0.99)
            repair_func: Optional function to repair infeasible solutions
            **particle_kwargs: Additional parameters passed to Particle initialization
        """
        self.num_particles = num_particles
        self.num_iter = num_iter
        self.num_dimensions = num_dimensions
        self.bounds = bounds
        self.compute_fitness = compute_fitness
        self.particle_type = particle_type
        self.repair_func = repair_func
        
        # PSO parameters
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.w_decay = w_decay
        
        # Initialize swarm
        self.particles = [
            Particle(num_dimensions, bounds, compute_fitness, 
                    particle_type, repair_func, **particle_kwargs)
            for _ in range(num_particles)
        ]
        
        # Global best
        self._update_global_best()
        self.fitness_history = []
    
    def _update_global_best(self):
        """Update global best from all particles"""
        best_particle = max(self.particles, key=lambda p: p.fitness)
        self.gbest_position = best_particle.position.copy()
        self.gbest_fitness = best_particle.fitness
    
    def run(self, verbose=True):
        """
        Run PSO optimization
        
        Args:
            verbose: Print progress every iteration
            
        Returns:
            (best_position, best_fitness)
        """
        for iteration in range(self.num_iter):
            # Update each particle
            for particle in self.particles:
                particle.update_velocity(self.gbest_position, self.w, self.c1, self.c2)
                particle.update_position()
                
                # Update global best if improved
                if particle.fitness > self.gbest_fitness:
                    self.gbest_fitness = particle.fitness
                    self.gbest_position = particle.position.copy()
            
            # Decay inertia weight
            self.w *= self.w_decay
            self.fitness_history.append(self.gbest_fitness)
            
            if verbose:
                print(f"Iteration {iteration+1}: best fitness = {self.gbest_fitness:.4f}")
        
        return self.gbest_position, self.gbest_fitness
