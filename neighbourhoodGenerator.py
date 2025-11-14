"""
Simple Neighborhood Generator Example

This module demonstrates a basic neighborhood generator implementation that
perturbs vessel trajectory features while maintaining domain constraints.

This is a standalone example/test script, not used in the main application.
It shows how to create synthetic vessel instances with realistic constraints
such as speed quartile ordering and log-transformed features.

Classes:
    NewGen: Basic perturbation generator for vessel features

Example usage is included at the bottom of the file.
"""

import random
import math

class NewGen:
    """
    Simple neighborhood generator for vessel trajectory features.
    
    This class generates perturbed instances of vessel data by randomly
    sampling from predefined ranges while maintaining constraints like
    speed quartile ordering.
    
    Attributes:
        gen_data: List of generated perturbed arrays
    """
    
    def __init__(self):
        """Initialize the generator with empty gen_data."""
        self.gen_data = None
 
    def perturb(self, x, num_instances):
        """
        Generate perturbed versions of a vessel instance.
        
        Creates synthetic instances by:
        1. Ensuring speed quartile ordering (min < Q1 < median < Q3)
        2. Sampling other features from their observed ranges
        3. Handling log-transformed features appropriately
        
        Note: This is a simplified version. The main application uses more
        sophisticated generators with better constraint enforcement.
        
        Args:
            x: Original instance array to perturb
            num_instances: Number of perturbed instances to generate
            
        Returns:
            List of perturbed instance arrays
        """
        perturbed_arrays = []
        for _ in range(num_instances):
            perturbed_arr = x[:]
 
            for val in range(0, int(len(x))):
                perturbed_arr[0] = random.uniform(0, 17.93)
                perturbed_arr[1] = random.uniform(perturbed_arr[0], 20.12) # always greater than minspeed
                perturbed_arr[2] = random.uniform(perturbed_arr[1], 20.75) # always greater than speedQ1
                perturbed_arr[3] = random.uniform(perturbed_arr[2], 21.65) # always greater than speedMedian
                perturbed_arr[4] = random.uniform(0, 2.24)
                perturbed_arr[5] = random.uniform(-0.24, 0.36)
                perturbed_arr[6] = random.uniform(-2.80, 1.77)
                perturbed_arr[7] = random.uniform(0.12, 282.26)
                perturbed_arr[8] = math.log10(random.uniform(math.exp(-3.05), perturbed_arr[7])) # inverse log transform, identify value less than max dist, then log transform again
 
            perturbed_arrays.append(perturbed_arr)
        self.gen_data = perturbed_arrays
        return perturbed_arrays

 
x = [0.03, 15.51, 15.91, 16.52, 0.004, 0.28, 0.98, 29.28, -1.86]
perturber = NewGen()
n = 10
perturbed_arrays = perturber.perturb(x=x, num_instances=n)
for perturbed_arr in perturbed_arrays:
    print(perturbed_arr)