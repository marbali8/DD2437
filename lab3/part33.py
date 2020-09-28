import numpy as np
import matplotlib.pyplot as plt

from train import hopfield

# exercice 1+2

x1 = np.array([-1, -1, 1, -1, 1, -1, -1, 1])
x2 = np.array([-1, -1, -1, -1, -1, 1, -1, -1])
x3 = np.array([-1, 1, 1, -1, -1, 1, -1, 1])
patterns = np.array([x1, x2, x3])

x1d = np.array([1, -1, 1, -1, 1, -1, -1, 1]) # 1 dif
x2d = np.array([1, 1, -1, -1, -1, 1, -1, -1]) # 2 dif
x3d = np.array([1, 1, 1, -1, 1, 1, -1, 1]) # 2 dif
patterns_d = np.array([x1d, x2d, x3d])

P = patterns_d.shape[0]
N = patterns_d.shape[1] # number of units

w = 1 / N * patterns.T @ patterns

all_combinations = [int(x) for n in np.arange(2**N) for x in format(n, '#0'+str(N+2)+'b')[2:]]
all_combinations = np.array(all_combinations).reshape(2**N, N)
all_combinations[all_combinations == 0] = -1

output = hopfield(patterns, all_combinations)
correct = np.sum(output == all_combinations, axis = 1) == N
attractors = all_combinations[np.where(correct)[0]]

plt.figure()
for i, p in enumerate(all_combinations):

    p = p.reshape((-1, 1))
    e = - np.sum(w * np.dot(p, p.T))
    color = 'orange'
    if np.any(np.sum(p.T == patterns_d, axis = 1) == N): # distorted
        color = 'green'
    elif np.any(np.sum(p.T == patterns, axis = 1) == N): # stored
        color = 'purple'
    elif np.any(i == np.where(correct)[0]): # attractors in general
        color = 'blue'
    plt.scatter([i], [e], c = color)

plt.xlabel('n pattern'), plt.ylabel('energy'), plt.show()

# the energy in the attractors is in majority twice as low as the others
# the energy in the distorted sequences is

# exercice 3

# is seq so i did it on matlab
