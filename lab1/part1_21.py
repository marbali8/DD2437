import numpy as np
import matplotlib.pyplot as plt

from delta import delta_rule_1hlayer_batch, delta_rule_1hlayer_batch_val
from extras import plot_error


## 3.2.1 Classification of linearly non-separable data

ndata = 100;
mA = [[-1.0], [0.3]]
sigmaA = 0.2
mB = [[0.0], [-0.1]]
sigmaB = 0.3

# np.random.seed(0)

classA = np.random.randn(2, ndata) * sigmaA + mA # (2, ndata)
classA[0, round(ndata/2):] = np.random.randn(1, round(ndata/2)) * sigmaA - mA[0]
classA = np.concatenate([classA, -np.ones((1, ndata))])  # (3, ndata)
classB = np.random.randn(2, ndata) * sigmaB + mB # (2, ndata)
classB = np.concatenate([classB, np.ones((1, ndata))])  # (3, ndata)

plt.scatter(classA[0], classA[1], c = 'red')
plt.scatter(classB[0], classB[1], c = 'blue')

## part 1

# data = np.concatenate([classA, classB], axis = 1) # (3, 2ndata)
# np.random.shuffle(data.T) # (3, 2ndata)
# patterns = data[:2, :] # (2, 2ndata)
# targets = data[-1:, :] # (1, 2ndata)
#
# e = delta_rule_1hlayer_batch(patterns, targets, epochs = 100, n_hidden = 100)
# plt.legend('-AB')
#
# plot_error(e, 'all train gen delta rule')
# plt.show()
# plt.close()

## part 2

training = np.concatenate([classA[:, round(ndata*0.75):], classB[:, round(ndata*0.75):]], axis = 1) # (3, 1.5*ndata)
validation = np.concatenate([classA[:, :round(ndata*0.25)], classB[:, :round(ndata*0.25)]], axis = 1)
np.random.shuffle(training.T) # (3, 1.5*ndata)
np.random.shuffle(validation.T)
patterns_tr = training[:2, :] # (2, 1.5*ndata)
patterns_val = validation[:2, :]
targets_tr = training[-1:, :] # (1, 1.5*ndata)
targets_te = validation[-1:, :]

e_tr, e_val = delta_rule_1hlayer_batch_val(patterns_tr, patterns_val, targets_tr, targets_te, epochs = 100, n_hidden = 10)
plt.legend('-AB')

plot_error(e_val)
plot_error(e_tr, '-25% each delta rule', new_fig = False)
plt.legend('vt')
plt.show()
plt.close()
