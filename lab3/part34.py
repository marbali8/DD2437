import numpy as np
import matplotlib.pyplot as plt

from train import hopfield

x1 = np.array([-1, -1, 1, -1, 1, -1, -1, 1])
x2 = np.array([-1, -1, -1, -1, -1, 1, -1, -1])
x3 = np.array([-1, 1, 1, -1, -1, 1, -1, 1])
patterns = np.array([x1, x2, x3])

P = patterns.shape[0]
N = patterns.shape[1] # number of units

distortion_type = False # distorted if false, noise added if true

patterns_d = patterns.copy().astype('float')
noise = np.arange(0, 1+0.1, 0.1)
correct_any = np.zeros((P, noise.shape[0]))
correct_all = np.zeros((P, noise.shape[0]))

for i, n in enumerate(noise):

    idx = np.random.randint(0, N, int(N*n))

    if distortion_type:

        patterns_d[:, idx] += np.random.rand(P, int(N*n))

    else:

        patterns_d[:, idx] *= -1

    output = hopfield(patterns, patterns_d)
    correct_any[:, i] = np.sum(output == patterns, axis = 1) / N
    correct_all[:, i] = correct_any[:, i] == 1

print(np.sum(correct_all, axis = 1))
plt.figure(), plt.plot(noise, correct_any[0, :]), plt.plot(noise, correct_any[1, :]), plt.plot(noise, correct_any[2, :])
plt.legend(['p1', 'p2', 'p3']), plt.xlabel('noise %'), plt.ylabel('% of correct bits')
plt.show()
