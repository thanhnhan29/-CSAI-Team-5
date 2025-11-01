from Problem import RF
from simulated_annealing import SimulatedAnnealing as SA
import numpy as np
import matplotlib.pyplot as  plt

a = RF(1,1.5)

test = SA(func=a, step_size=0.1, mode="min", seed=42)

x, fx, his = test.optimize((0,0), return_history=True)

# a.visualize(points=his)
print(a.global_minimum())


his = np.array(his)
plt.plot(a.loss(a(his[:,0], his[:,1])))

plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Loss over time")
plt.show()
