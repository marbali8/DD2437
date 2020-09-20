import numpy as np
import matplotlib.pyplot as plt

from rbf import delta_rbf

# 3.2 Regression with noise

## MU INITIALISATION
## sin(2x)

# x = np.arange(0, 2*np.pi+0.1, 0.1)
# patt_tr = x.copy()
# tar_tr = np.sin(2*patt_tr) + np.random.normal(0, 0.1, patt_tr.shape)
# patt_val = x[5:]
# tar_val = np.sin(2*patt_val) + np.random.normal(0, 0.1, patt_val.shape)
# n_hidden = 20
#
# plt.xlim(-1, np.max(patt_tr) + 1)
# y = np.ones(n_hidden)
#
# mu = np.linspace(np.min(patt_tr), np.max(patt_tr), n_hidden)
# out_val = delta_rbf(patt_tr, tar_tr, patt_val, tar_val, epochs = 120, eta = 0.0001, n_hidden = n_hidden, mu = mu, width = 0.01)
# print('equispaced {:.4f}'.format(np.mean(np.abs(out_val - tar_val))))
# plt.plot(mu, y*4, 'o', ms = 5)
#
# mu = np.random.rand(n_hidden)
# out_val = delta_rbf(patt_tr, tar_tr, patt_val, tar_val, epochs = 120, eta = 0.0001, n_hidden = n_hidden, mu = mu, width = 0.01)
# print('random {:.4f}'.format(np.mean(np.abs(out_val - tar_val))))
# plt.plot(mu, y*3, 'o', ms = 5)
#
# mu = np.random.rand(n_hidden)
# mu = mu / np.max(mu) * np.max(patt_tr)
# out_val = delta_rbf(patt_tr, tar_tr, patt_val, tar_val, epochs = 120, eta = 0.0001, n_hidden = n_hidden, mu = mu, width = 0.01)
# print('random scaled {:.4f}'.format(np.mean(np.abs(out_val - tar_val))))
# plt.plot(mu, y*2, 'o', ms = 5)
#
# mu = np.random.rand(n_hidden)
# mu = mu / np.max(mu) * np.max(patt_tr) - np.min(patt_tr)
# out_val = delta_rbf(patt_tr, tar_tr, patt_val, tar_val, epochs = 120, eta = 0.0001, n_hidden = n_hidden, mu = mu, width = 0.01)
# print('random scaled and shifted {:.4f}'.format(np.mean(np.abs(out_val - tar_val))))
# plt.plot(mu, y*1, 'o', ms = 5)
#
# plt.legend(['equispaced', 'random', 'random scaled', 'random scaled and shifted'])
# plt.axis('off')
# plt.show()


## box(2x)

x = np.arange(0, 2*np.pi+0.1, 0.1)
patt_tr = x.copy()
aux = np.sin(2*patt_tr) + np.random.normal(0, 0.1, patt_tr.shape)
tar_tr = 1*(aux >= 0) + (-1)*(aux < 0)
patt_val = x[5:]
aux = np.sin(2*patt_val) + np.random.normal(0, 0.1, patt_val.shape)
tar_val = 1*(aux >= 0) + (-1)*(aux < 0)
n_hidden = 20

plt.xlim(-1, np.max(patt_tr) + 1)
y = np.ones(n_hidden)

mu = np.linspace(np.min(patt_tr), np.max(patt_tr), n_hidden)
out_val = delta_rbf(patt_tr, tar_tr, patt_val, tar_val, epochs = 120, eta = 0.0001, n_hidden = n_hidden, mu = mu, width = 0.01)
print('equispaced {:.4f}'.format(np.mean(np.abs(out_val - tar_val))))
plt.plot(mu, y*4, 'o', ms = 5)

mu = np.random.rand(n_hidden)
out_val = delta_rbf(patt_tr, tar_tr, patt_val, tar_val, epochs = 120, eta = 0.0001, n_hidden = n_hidden, mu = mu, width = 0.01)
print('random {:.4f}'.format(np.mean(np.abs(out_val - tar_val))))
plt.plot(mu, y*3, 'o', ms = 5)

mu = np.random.rand(n_hidden)
mu = mu / np.max(mu) * np.max(patt_tr)
out_val = delta_rbf(patt_tr, tar_tr, patt_val, tar_val, epochs = 120, eta = 0.0001, n_hidden = n_hidden, mu = mu, width = 0.01)
print('random scaled {:.4f}'.format(np.mean(np.abs(out_val - tar_val))))
plt.plot(mu, y*2, 'o', ms = 5)

mu = np.random.rand(n_hidden)
mu = mu / np.max(mu) * np.max(patt_tr) - np.min(patt_tr)
out_val = delta_rbf(patt_tr, tar_tr, patt_val, tar_val, epochs = 120, eta = 0.0001, n_hidden = n_hidden, mu = mu, width = 0.01)
print('random scaled and shifted {:.4f}'.format(np.mean(np.abs(out_val - tar_val))))
plt.plot(mu, y*1, 'o', ms = 5)

plt.legend(['equispaced', 'random', 'random scaled', 'random scaled and shifted'])
plt.axis('off')
plt.show()
