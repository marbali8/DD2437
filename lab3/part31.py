import numpy as np

from train import hopfield

# 3.1 Convergence and attractors

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

# question 1

# output = hopfield(patterns, patterns_d)
# correct = np.sum(output == patterns, axis = 1) == N
#
# print(np.where(correct)[0])

# e = 2 to converge, x2d doesn't converge to stored pattern x2
# xid are not attractors (we can check by output == patterns_d)

# are x1, x2, x3 attractors?

# output = hopfield(patterns, patterns)
# correct = np.sum(output == patterns, axis = 1) == N
#
# print(np.where(correct)[0])

# e = 2 to converge, they are attractors bc recall(xi) = xi
# stored patterns are always attractors

# question 2

all_combinations = [int(x) for n in np.arange(2**N) for x in format(n, '#0'+str(N+2)+'b')[2:]]
all_combinations = np.array(all_combinations).reshape(2**N, N)
all_combinations[all_combinations == 0] = -1

output = hopfield(patterns, all_combinations)
correct = np.sum(output == all_combinations, axis = 1) == N
attractors = all_combinations[np.where(correct)[0]]

# prints attractor vectors and the index of x1, x2, x3 (as they were attractors too)
print(attractors)
print([np.where(np.sum(attractors == p, axis = 1) == N)[0] for p in patterns])

# question 3

# it sometimes converges into the correct one, sometimes doesn't, depends on if
# there is another attractor in the middle

# e = 2 to converge, log(N)
