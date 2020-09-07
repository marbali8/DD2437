import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from delta import delta_rule_0hlayer_batch
from extras import plot_error
from perceptron import perceptron_rule_0hlayer_batch

## 3.1.3 Classification of samples that are not linearly separable

### part 1

# ## create data
# n = 100;
# mA = [[-0.5], [0.75]]
# sigmaA = 0.5
# mB = [[0.75], [-0.5]]
# sigmaB = 0.5
#
# np.random.seed(0)
#
# classA = np.random.randn(2, n) * sigmaA + mA # (2, n)
# classA = np.concatenate([classA, -np.ones((1, n))])  # (3, n)
# classB = np.random.randn(2, n) * sigmaB + mB # (2, n)
# classB = np.concatenate([classB, np.ones((1, n))])  # (3, n)
#
# # make sure data is not linearly separable
# plt.scatter(classA[0], classA[1], c = 'red'),
# plt.scatter(classB[0], classB[1], c = 'blue')
# # plt.savefig('classes.png')
#
# data = np.concatenate([classA, classB], axis = 1) # (3, 2n)
# np.random.shuffle(data.T) # (3, 2n)
# patterns = data[:2, :] # (2, 2n)
# targets = data[-1:, :] # (1, 2n)
#
# ## compute error
# e_d = delta_rule_0hlayer_batch(patterns, targets)
# mse_p = perceptron_rule_0hlayer_batch(patterns, targets)
# plt.legend('dpAB')
#
# plot_error(e_d, 'delta rule')
# plot_error(mse_p, 'perceptron rule')
# plt.show()
# plt.close()

# we need to give it more epochs but it never converges.

### part 2

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
# plt.savefig('classes.png')

# TODO: represent only the training samples
## subsampling 1: all train

# data = np.concatenate([classA, classB], axis = 1) # (3, 2ndata)
# np.random.shuffle(data.T) # (3, 2ndata)
# patterns = data[:2, :] # (2, 2ndata)
# targets = data[-1:, :] # (1, 2ndata)
#
# e_d = delta_rule_0hlayer_batch(patterns, targets, epochs = 100)
# mse_p = perceptron_rule_0hlayer_batch(patterns, targets, epochs = 100)
# # c_max = confusion_matrix(targets, someoutput???, normalize = 'true')
# plt.legend('dpAB')
#
# plot_error(e_d, 'all train delta rule')
# plot_error(mse_p, 'all train perceptron rule')
# plt.show()
# plt.close()

## subsampling 2: -25% each

# data = np.concatenate([classA[:, round(ndata*0.75):], classB[:, round(ndata*0.75):]], axis = 1) # (3, 1.5*ndata)
# np.random.shuffle(data.T) # (3, 1.5*ndata)
# patterns = data[:2, :] # (2, 1.5*ndata)
# targets = data[-1:, :] # (1, 1.5*ndata)
#
# e_d = delta_rule_0hlayer_batch(patterns, targets, epochs = 100)
# mse_p = perceptron_rule_0hlayer_batch(patterns, targets, epochs = 100)
# plt.legend('dpAB')
#
# plot_error(e_d, '-25% each delta rule')
# plot_error(mse_p, '-25% each perceptron rule')
# plt.show()
# plt.close()

## subsampling 3: -50% A

# data = np.concatenate([classA[:, round(ndata*0.5):], classB], axis = 1) # (3, 1.5*ndata)
# np.random.shuffle(data.T) # (3, 1.5*ndata)
# patterns = data[:2, :] # (2, 1.5*ndata)
# targets = data[-1:, :] # (1, 1.5*ndata)
#
# e_d = delta_rule_0hlayer_batch(patterns, targets)
# mse_p = perceptron_rule_0hlayer_batch(patterns, targets)
# plt.legend('dpAB')
#
# plot_error(e_d, '-50% A delta rule')
# plot_error(mse_p, '-50% A perceptron rule')
# plt.show()
# plt.close()

## subsampling 4: -50% B

# data = np.concatenate([classA, classB[:, round(ndata*0.5):]], axis = 1) # (3, 1.5*ndata)
# np.random.shuffle(data.T) # (3, 1.5*ndata)
# patterns = data[:2, :] # (2, 1.5*ndata)
# targets = data[-1:, :] # (1, 1.5*ndata)
#
# e_d = delta_rule_0hlayer_batch(patterns, targets)
# mse_p = perceptron_rule_0hlayer_batch(patterns, targets)
# plt.legend('dpAB')
#
# plot_error(e_d, '-50% B delta rule')
# plot_error(mse_p, '-50% B perceptron rule')
# plt.show()
# plt.close()

## subsampling 5: -20% A[0, :]<0 and -80% A[0, :]>0

sub_A = classA[:, classA[0] < 0][:, round(ndata*0.8):]
sub_A = np.concatenate([sub_A, classA[:, classA[0] > 0][:, round(ndata*0.2):]], axis = 1)
data = np.concatenate([sub_A, classB], axis = 1) # (3, sub_A.shape[1])
np.random.shuffle(data.T) # (3, sub_A.shape[1])
patterns = data[:2, :] # (2, sub_A.shape[1])
targets = data[-1:, :] # (1, sub_A.shape[1])

e_d = delta_rule_0hlayer_batch(patterns, targets)
mse_p = perceptron_rule_0hlayer_batch(patterns, targets)
plt.legend('dpAB')

plot_error(e_d, '-20% A[0, :]<0 and -80% A[0, :]>0 delta rule')
plot_error(mse_p, '-20% A[0, :]<0 and -80% A[0, :]>0 perceptron rule')
plt.show()
plt.close()

# they all work like shit bc they need a curve! more parameters for the decision boundary, also, error perceptron up and down, error delta at least follows the -exp shape.
