import numpy as np
import matplotlib.pyplot as plt

from delta import delta_rule_0hlayer_batch, delta_rule_0hlayer_seq, delta_rule_0hlayer_batch_nbias
from extras import plot_error
from perceptron import perceptron_rule_0hlayer_batch, perceptron_rule_0hlayer_batch_nbias

## 3.1.2 Classification with a single-layer perceptron and analysis

## create data
n = 100;
mA = [[-1.5], [1.0]]
sigmaA = 0.5
mB = [[1.0], [-1.5]]
sigmaB = 0.5

# np.random.seed(0)

classA = np.random.randn(2, n) * sigmaA + mA # (2, n)
classA = np.concatenate([classA, -np.ones((1, n))])  # (3, n)
classB = np.random.randn(2, n) * sigmaB + mB # (2, n)
classB = np.concatenate([classB, np.ones((1, n))])  # (3, n)

# make sure data is linearly separable
plt.scatter(classA[0], classA[1], c = 'red', s = 2)
plt.scatter(classB[0], classB[1], c = 'blue', s = 2)
# plt.savefig('classes.png')

data = np.concatenate([classA, classB], axis = 1) # (3, 2n)
np.random.shuffle(data.T) # (3, 2n)
patterns = data[:2, :] # (2, 2n)
targets = data[-1:, :] # (1, 2n)

### question 1

# # compute error
# e_d = delta_rule_0hlayer_batch(patterns, targets, epochs = 30)
# mse_p = perceptron_rule_0hlayer_batch(patterns, targets, epochs = 30)
# plt.legend('dpAB')
#
# plot_error(e_d, 'delta rule')
# plot_error(mse_p, 'perceptron rule')
# plt.show()
# plt.close()

# 1. perceptron converges first, line is worse. low lr need more epochs.
# it can look like it doesn't but it's bc of the scatter point size

### question 2

e_b = delta_rule_0hlayer_batch(patterns, targets)
e_s = delta_rule_0hlayer_seq(patterns, targets)
plt.legend('bsAB')

plot_error(e_b, 'batch')
plot_error(e_s[::2*n].flatten(), 'sequential')
plt.show()
plt.close()

# 2. sequential converges last, very sensitive to random init! (comment seed(0))

### question 3
#
# e_nb = delta_rule_0hlayer_batch_nbias(patterns, targets)
# mse_p_nb = perceptron_rule_0hlayer_batch_nbias(patterns, targets)
# plt.legend('dpAB')
#
# plot_error(e_nb, 'no bias')
# plot_error(mse_p_nb, 'perceptron no bias')
# plt.show()
# plt.close()

# 3. it does converge...
