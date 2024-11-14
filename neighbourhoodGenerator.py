import random
import math

class NewGen:
    def __init__(self):
        self.gen_data = None
 
    def perturb(self, x, num_instances):
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