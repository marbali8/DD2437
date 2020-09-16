import numpy as np
import matplotlib.pyplot as plt

# 3.1 Batch more training using least squares - supervised learning of network weights

def least_squares_rbf(patterns_tr, targets_tr, patterns_val, targets_val, n_hidden = 10):

    ## initialisation
    ndata = patterns_tr.shape[0]

    mu = np.linspace(np.min(patterns_tr), np.max(patterns_tr), n_hidden)
    var = np.ones(n_hidden)*0.1
    w = np.random.randn(n_hidden)
    dw = np.zeros(w.shape)
    err = np.array([])

    ## forward pass
    phi = np.zeros((ndata, n_hidden))
    for n in range(n_hidden):

        phi[:, n] = np.exp(-(patterns_tr - mu[n])**2/(2*var[n]))
        # out = w[i]*oin

    w = np.linalg.solve(np.dot(phi.T, phi), np.dot(phi.T, targets_tr)) # eq (8)

    ## validation
    out_val = np.array([])
    for x in patterns_val:
        fhat = 0
        for n in range(n_hidden):
            fhat = fhat + w[n] * np.exp(-(x - mu[n])**2/(2*var[n]))
        out_val = np.append(out_val, fhat)

    return out_val

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
#     out_val = least_squares_rbf(patt_tr, tar_tr, patt_val, tar_val, n_hidden = n)
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

    out_val = np.sign(least_squares_rbf(patt_tr, tar_tr, patt_val, tar_val, n_hidden = n))
    this_err = np.mean(np.abs(out_val - tar_val))

    if this_err < err_val:
        err_val = err_val / 10 # 0.1, 0.01, 0.001
        plt.plot(patt_tr, out_val)
        print('val error at n_hidden = ' + str(n) + ': ' + '{:.4f}'.format(this_err))

    # if n % 50 == 0:
    #     print(str(n) + ' ' + '{:.4f}'.format(this_err))

    if n == 150:
        plt.plot(patt_tr, out_val)

    n = n + 1

plt.legend(['target', '<0.1', 'n=150'])
plt.show()

# 38, never?, never?
# with sign (added to out_val): 10, 15, 16
# we do sign bc we erase the oscillations, reminds me of fourier transform (it's literally it i think)
