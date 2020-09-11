import numpy as np
import matplotlib.pyplot as plt

from delta import delta_rule_1hlayer_batch, delta_rule_1hlayer_batch_val, delta_rule_1hlayer_seq_val
from extras import plot_error


## 3.2.1 Classification of linearly non-separable data

ndata = 100;
mA = [[-0.7], [0.3]]
sigmaA = 0.2
mB = [[0.0], [-0.3]]
sigmaB = 0.3

# np.random.seed(0)

classA = np.random.randn(2, ndata) * sigmaA + mA # (2, ndata)
classA[0, ::2] = np.random.randn(1, round(ndata/2)) * sigmaA - mA[0]
classA = np.concatenate([classA, -np.ones((1, ndata))])  # (3, ndata)
classB = np.random.randn(2, ndata) * sigmaB + mB # (2, ndata)
classB = np.concatenate([classB, np.ones((1, ndata))])  # (3, ndata)

plt.scatter(classA[0], classA[1], c = 'red', s = 2)
plt.scatter(classB[0], classB[1], c = 'blue', s = 2)

## part 1

# data = np.concatenate([classA, classB], axis = 1) # (3, 2ndata)
# np.random.shuffle(data.T) # (3, 2ndata)
# patterns = data[:2, :] # (2, 2ndata)
# targets = data[-1:, :] # (1, 2ndata)
#
# hidden = [5, 8, 11, 14]
#
# for h in hidden:
#     plt.figure(2)
#     e = delta_rule_1hlayer_batch(patterns, targets, epochs = 600, n_hidden = h, plot_acc = True)
#     print('e ' + str(e[-1]))
#
# plt.figure(1), plt.legend(['5 hidden', '8 hidden', '11 hidden', '14 hidden', 'A', 'B'])
# plt.show()
# plt.close()

## part 2 (subsampling 1)

# training = np.concatenate([classA[:, :round(ndata*0.75)], classB[:, :round(ndata*0.75)]], axis = 1) # (3, 1.5*ndata)
# validation = np.concatenate([classA[:, round(ndata*0.75):], classB[:, round(ndata*0.75):]], axis = 1)
# np.random.shuffle(training.T) # (3, 1.5*ndata)
# np.random.shuffle(validation.T)
# patterns_tr = training[:2, :] # (2, 1.5*ndata)
# patterns_val = validation[:2, :]
# targets_tr = training[-1:, :] # (1, 1.5*ndata)
# targets_te = validation[-1:, :]
#
# plt.scatter(classA[0, round(ndata*0.75):], classA[1, round(ndata*0.75):], c = 'red', s = 9)
# plt.scatter(classB[0, round(ndata*0.75):], classB[1, round(ndata*0.75):], c = 'blue', s = 9)
#
# hidden = [1, 4, 8, 13]
#
# for h in hidden:
#     print(h)
#     e_tr, e_val = delta_rule_1hlayer_batch_val(patterns_tr, patterns_val, targets_tr, targets_te, epochs = 500, n_hidden = h, eta = 0.01)
#     print('e_val ' + str(e_val[-1]))
#     print('e_tr ' + str(e_tr[-1]))
#
# plt.figure(1), plt.legend(['5 hidden', '8 hidden', '11 hidden', '14 hidden', 'A', 'B'])
# plt.show()
# plt.close()

## part 2 (subsampling 3)

training = np.concatenate([classA[:, :round(ndata*0.5)], classB[:, :round(ndata*1.0)]], axis = 1) # (3, 1.5*ndata)
validation = classA[:, round(ndata*0.5):]
np.random.shuffle(training.T) # (3, 1.5*ndata)
np.random.shuffle(validation.T)
patterns_tr = training[:2, :] # (2, 1.5*ndata)
patterns_val = validation[:2, :]
targets_tr = training[-1:, :] # (1, 1.5*ndata)
targets_te = validation[-1:, :]

plt.scatter(classA[0, round(ndata*0.5):], classA[1, round(ndata*0.5):], c = 'red', s = 9)
# plt.scatter(classB[0, :round(ndata*0.0)], classB[1, :round(ndata*0.0)], c = 'blue', s = 9)

hidden = [1, 4, 8, 13]

for h in hidden:
    print(h)
    e_tr, e_val = delta_rule_1hlayer_batch_val(patterns_tr, patterns_val, targets_tr, targets_te, epochs = 500, n_hidden = h, eta = 0.01)
    print('e_val ' + str(e_val[-1]))
    print('e_tr ' + str(e_tr[-1]))

plt.figure(1), plt.legend(['5 hidden', '8 hidden', '11 hidden', '14 hidden', 'A', 'B'])
plt.show()
plt.close()



# plot_error(e_val)
# plot_error(e_tr, 'batch -25% each delta rule', new_fig = False)
# plt.legend('vt')
# print('e_val ' + str(e_val[-1]))
#
# plot_error(e_s_val[::patterns_val.shape[1]].flatten())
# plot_error(e_s_tr[::patterns_val.shape[1]].flatten(), 'seq -25% each delta rule', new_fig = False)
# plt.legend('vt')
# print('e_s_val ' + str(e_s_val[-1]))

# plt.show()
# plt.close()
