import numpy as np
import matplotlib.pyplot as plt

from rbf import delta_rbf

# 3.2 Regression with noise
## LEARNING RATE
## sin(2x)

# x = np.arange(0, 2*np.pi+0.1, 0.1)
# patt_tr = x.copy()
# tar_tr = np.sin(2*patt_tr) + np.random.normal(0, 0.1, patt_tr.shape)
# patt_val = x[5:]
# tar_val = np.sin(2*patt_val) + np.random.normal(0, 0.1, patt_val.shape)
#
# lr = [0.0001, 0.001, 0.01]
#
# for eta in lr:
#
#     out_val = delta_rbf(patt_tr, tar_tr, patt_val, tar_val, eta = eta, epochs = 120, n_hidden = 20, width = 0.01, fig_err = 1)
#
# plt.legend(lr)
# plt.show()

## box(2x)

x = np.arange(0, 2*np.pi+0.1, 0.1)
patt_tr = x.copy()
aux = np.sin(2*patt_tr) + np.random.normal(0, 0.1, patt_tr.shape)
tar_tr = 1*(aux >= 0) + (-1)*(aux < 0)
patt_val = x[5:]
aux = np.sin(2*patt_val) + np.random.normal(0, 0.1, patt_val.shape)
tar_val = 1*(aux >= 0) + (-1)*(aux < 0)

lr = [0.0001, 0.001, 0.01]

for eta in lr:

    out_val = delta_rbf(patt_tr, tar_tr, patt_val, tar_val, eta = eta, epochs = 1200, n_hidden = 20, width = 0.01, fig_err = 1)

plt.legend(lr)
plt.show()
