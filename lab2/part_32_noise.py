import numpy as np
import matplotlib.pyplot as plt

from rbf import batch_rbf, delta_rbf

# 3.2 Regression with noise
# VALIDATION WITH AND WITHOUT ERROR

## sin(2x)

# x = np.arange(0, 2*np.pi+0.1, 0.1)
# patt_tr = x.copy()
# patt_tr_n = patt_tr + np.random.normal(0, 0.1, x.shape)
# tar_tr = np.sin(2*patt_tr)
# tar_tr_n = np.sin(2*patt_tr_n)
#
# patt_val = x[5:]
# patt_val_n = patt_val + np.random.normal(0, 0.1, patt_val.shape)
# tar_val = np.sin(2*patt_val)
# tar_val_n = np.sin(2*patt_val_n)
#
# out_val = delta_rbf(patt_tr, tar_tr, patt_val, tar_val, eta = 0.0001, epochs = 120, n_hidden = 20, width = 0.01)
# err = np.mean(np.abs(out_val - tar_val))
# out_val_ntr = delta_rbf(patt_tr_n, tar_tr_n, patt_val, tar_val, eta = 0.0001, epochs = 120, n_hidden = 20, width = 0.01)
# err_ntr = np.mean(np.abs(out_val_ntr - tar_val))
# out_val_ntrte = delta_rbf(patt_tr_n, tar_tr_n, patt_val_n, tar_val_n, eta = 0.0001, epochs = 120, n_hidden = 20, width = 0.01)
# err_ntrte = np.mean(np.abs(out_val_ntrte - tar_val_n))
#
# plt.figure(), plt.title('sin delta test outputs')
# plt.plot(patt_tr, tar_tr)
# plt.plot(patt_val, out_val)
# plt.plot(patt_val, out_val_ntr)
# plt.plot(patt_val_n, out_val_ntrte)
# plt.legend(['target', 'no noise (err {:.4f})'.format(err), 'noise in train (err {:.4f})'.format(err_ntr), 'noise in both (err {:.4f})'.format(err_ntrte)])
#
#
# out_val = batch_rbf(patt_tr, tar_tr, patt_val, tar_val, n_hidden = 20, width = 0.01)
# err = np.mean(np.abs(out_val - tar_val))
# out_val_ntr = batch_rbf(patt_tr_n, tar_tr_n, patt_val, tar_val, n_hidden = 20, width = 0.01)
# err_ntr = np.mean(np.abs(out_val_ntr - tar_val))
# out_val_ntrte = batch_rbf(patt_tr_n, tar_tr_n, patt_val_n, tar_val_n, n_hidden = 20, width = 0.01)
# err_ntrte = np.mean(np.abs(out_val_ntrte - tar_val_n))
#
# plt.figure(), plt.title('sin batch test outputs')
# plt.plot(patt_tr, tar_tr)
# plt.plot(patt_val, out_val)
# plt.plot(patt_val, out_val_ntr)
# plt.plot(patt_val_n, out_val_ntrte)
# plt.legend(['target', 'no noise (err {:.4f})'.format(err), 'noise in train (err {:.4f})'.format(err_ntr), 'noise in both (err {:.4f})'.format(err_ntrte)])
#
# plt.show()

## box(2x)

x = np.arange(0, 2*np.pi+0.1, 0.1)
patt_tr = x.copy()
patt_tr_n = patt_tr + np.random.normal(0, 0.1, x.shape)
aux = np.sin(2*patt_tr)
tar_tr = 1*(aux >= 0) + (-1)*(aux < 0)
aux = np.sin(2*patt_tr_n)
tar_tr_n = 1*(aux >= 0) + (-1)*(aux < 0)

patt_val = x[5:]
patt_val_n = patt_val + np.random.normal(0, 0.1, patt_val.shape)
aux = np.sin(2*patt_val)
tar_val = 1*(aux >= 0) + (-1)*(aux < 0)
aux = np.sin(2*patt_val_n)
tar_val_n = 1*(aux >= 0) + (-1)*(aux < 0)

out_val = delta_rbf(patt_tr, tar_tr, patt_val, tar_val, eta = 0.0001, epochs = 120, n_hidden = 20, width = 0.01)
err = np.mean(np.abs(out_val - tar_val))
out_val_ntr = delta_rbf(patt_tr_n, tar_tr_n, patt_val, tar_val, eta = 0.0001, epochs = 120, n_hidden = 20, width = 0.01)
err_ntr = np.mean(np.abs(out_val_ntr - tar_val))
out_val_ntrte = delta_rbf(patt_tr_n, tar_tr_n, patt_val_n, tar_val_n, eta = 0.0001, epochs = 120, n_hidden = 20, width = 0.01)
err_ntrte = np.mean(np.abs(out_val_ntrte - tar_val_n))

plt.figure(), plt.title('box delta test outputs')
plt.plot(patt_tr, tar_tr)
plt.plot(patt_val, out_val)
plt.plot(patt_val, out_val_ntr)
plt.plot(patt_val_n, out_val_ntrte)
plt.legend(['target', 'no noise (err {:.4f})'.format(err), 'noise in train (err {:.4f})'.format(err_ntr), 'noise in both (err {:.4f})'.format(err_ntrte)])


out_val = batch_rbf(patt_tr, tar_tr, patt_val, tar_val, n_hidden = 20, width = 0.01)
err = np.mean(np.abs(out_val - tar_val))
out_val_ntr = batch_rbf(patt_tr_n, tar_tr_n, patt_val, tar_val, n_hidden = 20, width = 0.01)
err_ntr = np.mean(np.abs(out_val_ntr - tar_val))
out_val_ntrte = batch_rbf(patt_tr_n, tar_tr_n, patt_val_n, tar_val_n, n_hidden = 20, width = 0.01)
err_ntrte = np.mean(np.abs(out_val_ntrte - tar_val_n))

plt.figure(), plt.title('box batch test outputs')
plt.plot(patt_tr, tar_tr)
plt.plot(patt_val, out_val)
plt.plot(patt_val, out_val_ntr)
plt.plot(patt_val_n, out_val_ntrte)
plt.legend(['target', 'no noise (err {:.4f})'.format(err), 'noise in train (err {:.4f})'.format(err_ntr), 'noise in both (err {:.4f})'.format(err_ntrte)])

plt.show()
