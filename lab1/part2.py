import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

from extras import plot_error

x = np.array([1.25]) # x(0) = x[0]
for t in range(1500+5): # [0, 1504]
    if t < 25:
        x = np.append(x, 0.9*x[t])
    else:
        x = np.append(x, x[t] + (0.2 * x[t-25]) / (1 + x[t-25]**10) - 0.1 * x[t])

# plt.plot(x, linewidth = 1)
# plt.title('Mackey-Glass time series')

patterns = np.zeros((1200, 5))
targets = np.zeros((1200,))
for t in range(301, 1500+1):
    patterns[t-301] = np.array([x[t-20], x[t-15], x[t-10], x[t-5], x[t]])
    targets[t-301] = x[t+5]

patterns_tr, patterns_te, targets_tr, targets_te = train_test_split(patterns, targets, test_size = 0.3, shuffle = False)

## part 1

# # 3 layer mlp (without output activation so that targets can be continuous for regression)
# nn = MLPRegressor(hidden_layer_sizes = (20, 10), early_stopping = True) # learning_rate_init = 0.0001, validation_fraction = 0.1)
# # solver sgd and activation logistic literally fuck it up the most (will not edit momentum bc it's only used with solver sgd)
# # early_stopping works!
# # i feel like alpha (regularisation) doesn't really do much. the most is hidden layers and early_stopping
# nn.fit(patterns_tr, targets_tr)
#
# out_te = nn.predict(patterns_te)
# plt.figure()
# plt.plot(targets_te)
# plt.plot(out_te)
# mse = ((out_te - targets_te)**2).mean()
# plt.legend(['target', 'predicted (mse ' + '{:.4f}'.format(mse) + ')'])
# plt.show()
# plt.close()

## part 2

std = [0.03, 0.09, 0.18]
h_size = [(20, 4), (20, 6), (20, 8), (20, 10)]
alpha = np.arange(2, 6)
idx = [(s, h) for s in range(len(std)) for h in range(len(h_size))]

for ai, a in enumerate(alpha):

    weights = np.array([])

    for i in idx:

        patterns_tr += np.random.normal(0, std[i[0]], patterns_tr.shape)
        patterns_te += np.random.normal(0, std[i[0]], patterns_te.shape)
        targets_tr += np.random.normal(0, std[i[0]], targets_tr.shape)
        targets_te += np.random.normal(0, std[i[0]], targets_te.shape)

        nn = MLPRegressor(hidden_layer_sizes = h_size[i[1]], alpha = 1/(10**a))
        nn.fit(patterns_tr, targets_tr)
        out_te = nn.predict(patterns_te)
        print(a, std[i[0]], h_size[i[1]], '{:.4f}'.format(((out_te - targets_te)**2).mean()))

        # weights size = in_dim*hidden_neurons_1 + hidden_neurons_1*hidden_neurons_2
        weights = np.concatenate((weights, nn.coefs_[0].flatten(), nn.coefs_[1].flatten()))

    plt.figure()
    plt.hist(weights, 25, rwidth = 0.9)
    plt.title('Weight histogram for a = 1/10**' + str(int(a)))

plt.show()
