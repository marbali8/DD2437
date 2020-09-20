import numpy as np
import matplotlib.pyplot as plt

from rbf import batch_rbf, delta_rbf

# 3.2 Regression with noise

## COMPARISON FOR N_HIDDEN AND WIDTHS
## sin(2x)

x = np.arange(0, 2*np.pi+0.1, 0.1)
patt_tr = x.copy()
tar_tr = np.sin(2*patt_tr) + np.random.normal(0, 0.1, patt_tr.shape)
patt_val = x[5:]
tar_val = np.sin(2*patt_val) + np.random.normal(0, 0.1, patt_val.shape)

widths = [0.0001, 0.001, 0.01, 0.1, 1]
nodes = [5, 10, 15, 20, 25, 30]
values = [(nodes[n], widths[w]) for n in range(len(nodes)) for w in range(len(widths))]

for (n, w) in values:

    out_val = delta_rbf(patt_tr, tar_tr, patt_val, tar_val, eta = 0.005, epochs = 20, n_hidden = n, width = w)
    out_val_b = batch_rbf(patt_tr, tar_tr, patt_val, tar_val, n_hidden = n, width = w)
    err = np.sqrt(np.mean((out_val - tar_val)**2))
    err_b = np.sqrt(np.mean((out_val_b - tar_val)**2))
    plt.figure(2), plt.scatter(n, w, s = int(err*100)+1, c = 'blue'), plt.annotate('{:.4f}'.format(err), (n, w))
    plt.figure(3), plt.scatter(n, w, s = int(err_b*100)+1, c = 'blue'), plt.annotate('{:.4f}'.format(err_b), (n, w))

plt.figure(2), plt.yscale('log'), plt.ylim([1e-5, 10]), plt.xlabel('nodes'), plt.ylabel('widths'), plt.title('delta rule')
plt.figure(3), plt.yscale('log'), plt.ylim([1e-5, 10]), plt.xlabel('nodes'), plt.ylabel('widths'), plt.title('batch rule')
# plt.xticks(range(len(values)), labels)
plt.show()

## box(2x)

# x = np.arange(0, 2*np.pi+0.1, 0.1)
# patt_tr = x.copy()
# aux = np.sin(2*patt_tr) + np.random.normal(0, 0.1, patt_tr.shape)
# tar_tr = 1*(aux >= 0) + (-1)*(aux < 0)
# patt_val = x[5:]
# aux = np.sin(2*patt_val) + np.random.normal(0, 0.1, patt_val.shape)
# tar_val = 1*(aux >= 0) + (-1)*(aux < 0)
#
# widths = [0.001, 0.01, 0.1, 0.5]
# nodes = [10, 20, 30, 40]
# values = [(nodes[n], widths[w]) for n in range(len(nodes)) for w in range(len(widths))]
#
# for (n, w) in values:
#
#     out_val = delta_rbf(patt_tr, tar_tr, patt_val, tar_val, eta = 0.005, epochs = 20, n_hidden = n, width = w)
#     out_val_b = np.sign(batch_rbf(patt_tr, tar_tr, patt_val, tar_val, n_hidden = n, width = w))
#     err = np.sqrt(np.mean((out_val - tar_val)**2))
#     err_b = np.sqrt(np.mean((out_val_b - tar_val)**2))
#     plt.figure(1), plt.scatter(n, w, s = err*100, c = 'blue'), plt.annotate('{:.4f}'.format(err), (n, w))
#     plt.figure(2), plt.scatter(n, w, s = err_b*100, c = 'blue'), plt.annotate('{:.4f}'.format(err_b), (n, w))
#
# plt.figure(1), plt.yscale('log'), plt.ylim([1e-4, 10]), plt.xlabel('nodes'), plt.ylabel('widths'), plt.title('delta rule')
# plt.figure(2), plt.yscale('log'), plt.ylim([1e-4, 10]), plt.xlabel('nodes'), plt.ylabel('widths'), plt.title('batch rule (with sign)')
# # plt.xticks(range(len(values)), labels)
# plt.show()
