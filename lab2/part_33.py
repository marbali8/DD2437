import numpy as np
import matplotlib.pyplot as plt

from rbf import delta_rbf, delta_rbf_cl

# 3.3 Competitive learning (CL) to initialise RBF units

## sin(2x)

x = np.arange(0, 2*np.pi+0.1, 0.1)
patt_tr = x.copy()
patt_tr_n = patt_tr + np.random.normal(0, 0.1, x.shape)
tar_tr = np.sin(2*patt_tr)
tar_tr_n = np.sin(2*patt_tr_n)

patt_val = x[5:]
patt_val_n = patt_val + np.random.normal(0, 0.1, patt_val.shape)
tar_val = np.sin(2*patt_val)
tar_val_n = np.sin(2*patt_val_n)

# best 3.1
out_val_1 = delta_rbf(patt_tr, tar_tr, patt_val, tar_val, eta = 0.0005, epochs = 40, n_hidden = 4, fig_err = 1, fig_mu = 2)

# best 3.2
out_val_2 = delta_rbf(patt_tr_n, tar_tr_n, patt_val_n, tar_val_n, eta = 0.0005, epochs = 40, n_hidden = 4, fig_err = 1, fig_mu = 2)

out_val_3 = delta_rbf_cl(patt_tr, tar_tr, patt_val, tar_val, eta = 0.0005, epochs = 40, n_hidden = 4, nhood = 1, fig_err = 1, fig_mu = 2)
out_val_4 = delta_rbf_cl(patt_tr, tar_tr, patt_val, tar_val, eta = 0.0005, epochs = 40, n_hidden = 4, nhood = 2, fig_err = 1, fig_mu = 2)

plt.figure(1), plt.title('validation error'), plt.legend(['no noise', 'noise', 'rbf', 'rbf dead'])
plt.figure(2), plt.title('mu'), plt.legend(['no noise', 'noise', 'rbf', 'rbf dead'])
plt.figure(3), plt.plot(tar_val), plt.plot(out_val_1), plt.plot(out_val_2), plt.plot(out_val_3), plt.plot(out_val_4)
plt.title('output'), plt.legend(['target', 'no noise', 'noise', 'rbf', 'rbf dead'])

plt.show()
