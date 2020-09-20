import numpy as np
import matplotlib.pyplot as plt

from rbf import delta_rbf, delta_rbf_cl

# 3.3 Competitive learning (CL) to initialise RBF units

## sin(2x)

x = np.arange(0, 2*np.pi+0.1, 0.1)
xrand = np.random.permutation(x)
patt_tr = xrand[:int(x.size*0.8)]
tar_tr = np.sin(2*patt_tr)
tar_tr_n = np.sin(2*patt_tr) + np.random.normal(0, 0.1, patt_tr.shape)

patt_val = xrand[int(x.size*0.8):]
tar_val = np.sin(2*patt_val)
tar_val_n = np.sin(2*patt_val) + np.random.normal(0, 0.1, patt_val.shape)

# best 3.1
out_val_1 = delta_rbf(patt_tr, tar_tr, patt_val, tar_val, eta = 0.005, epochs = 500, n_hidden = 30, width = 1e-1, fig_err = 1, fig_mu = 2)

# best 3.2
out_val_2 = delta_rbf(patt_tr, tar_tr_n, patt_val, tar_val_n, eta = 0.005, epochs = 500, n_hidden = 30, width = 1e-2, fig_err = 1, fig_mu = 2)

out_val_3 = delta_rbf_cl(patt_tr, tar_tr, patt_val, tar_val, eta = 0.005, epochs = 500, n_hidden = 30, nhood = 1, fig_err = 1, fig_mu = 2)
out_val_4 = delta_rbf_cl(patt_tr, tar_tr, patt_val, tar_val, eta = 0.005, epochs = 500, n_hidden = 30, nhood = 5, fig_err = 1, fig_mu = 2)

plt.figure(1), plt.title('validation error'), plt.legend(['no noise', 'noise', 'rbf', 'rbf dead'])
plt.figure(2), plt.title('mu'), plt.legend(['no noise', 'noise', 'rbf', 'rbf dead'])
plt.figure(3), plt.plot(tar_tr, linewidth = 2), plt.plot(out_val_1, linewidth = 0.5), plt.plot(out_val_2, linewidth = 0.5), plt.plot(out_val_3, linewidth = 3), plt.plot(out_val_4, linewidth = 3)
plt.title('output'), plt.legend(['target', 'no noise', 'noise', 'rbf', 'rbf dead'])

plt.show()

# randomising x (xrand) really helps with the validation!
# it really depends on the data sometimes (rbf)
# with noise is always the worse
# more nodes don't help with generalisation
# the position of the nodes is actually not that different xd
# rfb converges but eta really affects, like it's not completely smooth
