import numpy as np
import random

class Particle:
    def __init__(self, num_dimensions, bounds, compute_fitness, 
                 particle_type='continuous', repair_func=None,
                 init_position_ratio=0.1, init_velocity_ratio=0.1, 
                 velocity_range=(-2, 2)):
        """
        Initialize a particle for PSO
        
        Args:
            num_dimensions: Number of dimensions in the search space
            bounds: List of tuples [(min, max), ...] for each dimension
            compute_fitness: Function to compute fitness of a position
            particle_type: 'continuous' or 'binary'
            repair_func: Optional function to repair infeasible solutions
            init_position_ratio: For binary: probability of selecting (0.1 = 10%)
                                For continuous: not used (random in bounds)
            init_velocity_ratio: For continuous: initial velocity as ratio of bounds (0.1 = 10%)
            velocity_range: For binary: initial velocity range tuple (min, max)
        """
        self.num_dimensions = num_dimensions
        self.bounds = bounds
        self.compute_fitness = compute_fitness
        self.particle_type = particle_type
        self.repair_func = repair_func
        
        # Initialize position
        if particle_type == 'continuous':
            self.position = np.array([
                random.uniform(bounds[i][0], bounds[i][1]) 
                for i in range(num_dimensions)
            ])
        else:  # binary
            self.position = (np.random.random(num_dimensions) < init_position_ratio).astype(int)
            if self.repair_func:
                self.position = self.repair_func(self.position)
        
        # Initialize velocity
        if particle_type == 'continuous':
            v_range = [(bounds[i][1] - bounds[i][0]) * init_velocity_ratio 
                      for i in range(num_dimensions)]
            self.velocity = np.array([
                random.uniform(-v_range[i], v_range[i])
                for i in range(num_dimensions)
            ])
        else:  # binary
            self.velocity = np.random.uniform(velocity_range[0], velocity_range[1], 
                                             size=num_dimensions)
        
        # Fitness and personal best
        self.fitness = self.compute_fitness(self.position)
        self.pbest_position = self.position.copy()
        self.pbest_fitness = self.fitness
        
    def update_velocity(self, gbest_position, w, c1, c2, velocity_clamp_ratio=0.2):
        """
        Update particle velocity using PSO formula: v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
        
        Args:
            gbest_position: Global best position
            w: Inertia weight
            c1: Cognitive parameter (attraction to personal best)
            c2: Social parameter (attraction to global best)
            velocity_clamp_ratio: For continuous: max velocity as ratio of bounds
        """
        r1 = np.random.random(self.num_dimensions)
        r2 = np.random.random(self.num_dimensions)
        
        cognitive = c1 * r1 * (self.pbest_position - self.position)
        social = c2 * r2 * (gbest_position - self.position)
        self.velocity = w * self.velocity + cognitive + social
        
        # Velocity clamping for continuous problems only
        if self.particle_type == 'continuous':
            for i in range(self.num_dimensions):
                v_max = (self.bounds[i][1] - self.bounds[i][0]) * velocity_clamp_ratio
                self.velocity[i] = np.clip(self.velocity[i], -v_max, v_max)
    
    def update_position(self):
        """Update particle position: continuous uses x+v, binary uses sigmoid transfer"""
        if self.particle_type == 'continuous':
            # Update position and clip to bounds
            self.position = self.position + self.velocity
            for i in range(self.num_dimensions):
                self.position[i] = np.clip(self.position[i], 
                                          self.bounds[i][0], 
                                          self.bounds[i][1])
        else:  # binary: sigmoid transfer function
            sigmoid = 1 / (1 + np.exp(-self.velocity))
            self.position = (np.random.random(self.num_dimensions) < sigmoid).astype(int)
            if self.repair_func:
                self.position = self.repair_func(self.position)
        
        # Update fitness and personal best
        self.fitness = self.compute_fitness(self.position)
        if self.fitness > self.pbest_fitness:
            self.pbest_fitness = self.fitness
            self.pbest_position = self.position.copy()
