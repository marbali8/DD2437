import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


from extras import plot_error

x = np.array([1.25]) # x(0) = x[0]
for t in range(1500+5): # [0, 1504]
    if t < 25:
        x = np.append(x, 0.9*x[t])
    else:
        x = np.append(x, x[t] + (0.2 * x[t-25]) / (1 + x[t-25]**10) - 0.1 * x[t])

patterns = np.zeros((5, 1200))
targets = np.zeros((1, 1200))
for t in range(301, 1500+1):
    patterns[:, t-301] = np.array([x[t-20], x[t-15], x[t-10], x[t-5], x[t]]).T
    targets[:, t-301] = x[t+5]

patterns_tr, patterns_te, targets_tr, targets_te = train_test_split(patterns.T, targets.T, test_size = 0.3)

nn = MLPClassifier(hidden_layer_sizes = (10,), activation = 'logistic', solver = 'sgd', learning_rate_init = 0.001, momentum = 0.9, early_stopping = True, validation_fraction = 0.1)
nn.fit(patterns_tr.T, targets_tr.T)

out_te = nn.predict(targets_te)
mse = ((out_te - targets_te)**2).mean()
plot_error(e_tr)
plt.show()
plt.close()
