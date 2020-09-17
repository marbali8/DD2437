import numpy as np
import matplotlib.pyplot as plt

from rbf import batch_rbf

# 3.1 Batch more training using least squares - supervised learning of network weights

x = np.arange(0, 2*np.pi+0.1, 0.1)
a = np.sin(2*x)
b = 1*(a >= 0) + (-1)*(a < 0)

# sin(2x)

# patt_tr = x.copy()
# tar_tr = a.copy()
# patt_val = x.copy()
# tar_val = a.copy()
#
# plt.plot(patt_tr, tar_tr, linewidth = 1)
# n = 10
# err_val = 0.1
#
# while err_val >= 0.001:
#
#     out_val = batch_rbf(patt_tr, tar_tr, patt_val, tar_val, n_hidden = n)
#     this_err = np.mean(np.abs(out_val - tar_val))
#
#     if this_err < err_val:
#         err_val = err_val / 10 # 0.1, 0.01, 0.001
#         plt.plot(patt_tr, out_val)
#         print('val error at n_hidden = ' + str(n) + ': ' + '{:.4f}'.format(this_err))
#
#     n = n + 1
#
# plt.legend(['target', '<0.1', '<0.01', '<0.001'])
# plt.show()

# 10, 13, 29

# box(sin)

patt_tr = x.copy()
tar_tr = b.copy()
patt_val = x.copy()
tar_val = b.copy()

plt.plot(patt_tr, tar_tr, linewidth = 1)
n = 10
err_val = 0.1

while err_val >= 0.001 and n < 151:

    out_val = np.sign(batch_rbf(patt_tr, tar_tr, patt_val, tar_val, n_hidden = n))
    this_err = np.mean(np.abs(out_val - tar_val))

    if this_err < err_val:
        err_val = err_val / 10 # 0.1, 0.01, 0.001
        plt.plot(patt_val, out_val)
        print('val error at n_hidden = ' + str(n) + ': ' + '{:.4f}'.format(this_err))

    # if n % 50 == 0:
    #     print(str(n) + ' ' + '{:.4f}'.format(this_err))

    if n == 150:
        plt.plot(patt_val, out_val)

    n = n + 1

plt.legend(['target', '<0.1', 'n=150'])
plt.show()

# 38, never?, never?
# with sign (added to out_val): 10, 15, 16
# we do sign bc we erase the oscillations, reminds me of fourier transform (it's literally it i think)
