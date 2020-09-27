import numpy as np

x1 = np.array([-1, -1, 1, -1, 1, -1, -1, 1])
x2 = np.array([-1, -1, -1, -1, -1, 1, -1, -1])
x3 = np.array([-1, 1, 1, -1, -1, 1, -1, 1])
patterns = np.array([x1, x2, x3])

P = patterns.shape[0]
N = patterns.shape[1] # number of units

# hebbian learning

# w = np.zeros((N, N))
# for i in np.arange(N):
#
#     for j in np.arange(N):
#
#         w[i, j] = 1 / N * patterns[:, i].T @ patterns[:, j] # (P,) x (P,) = ()
w = 1 / N * patterns.T @ patterns # (N, N), normalisation generally not needed

# hopfield network recall

mult = patterns @ w
patterns = 1*(mult >= 0) - 1*(mult < 0)
