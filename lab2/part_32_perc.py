import numpy as np
import matplotlib.pyplot as plt
import time

from rbf import batch_rbf

# 3.2 Regression with noise
# COMPARISON WITH PERCEPTRON

def delta_1hlayer(p_tr, p_val, t_tr, t_val, n_hidden, eta = 0.001, epochs = 200, alpha = 0.9, fig_err = None):

    ## initialisation
    p_tr = p_tr.reshape((1, -1))
    p_val = p_val.reshape((1, -1))
    ndata = p_tr.shape[1] # cols

    v = np.random.randn(n_hidden, 1) # from input to hidden
    w = np.random.randn(1, n_hidden) # from hidden to output
    dw = np.zeros(w.shape)
    dv = np.zeros(v.shape)
    err_val = np.array([])

    for e in range(epochs): # [0, epochs-1]

        ## forward pass (layer for layer, from start to end)
        hin = np.dot(v, p_tr)
        hout = 2 / (1 + np.exp(-hin)) - 1
        oin = np.dot(w, hout)
        out = 2 / (1 + np.exp(-oin)) - 1

        ## validation
        hin_val = np.dot(v, p_val)
        hout_val = 2 / (1 + np.exp(-hin_val)) - 1
        oin_val = np.dot(w, hout_val)
        out_val = 2 / (1 + np.exp(-oin_val)) - 1
        th_out = -1*(out_val <= 0) + 1*(out_val > 0)

        err_val = np.append(err_val, np.sqrt(np.mean((th_out - tar_val)**2)))

        ## backward pass
        delta_o = (out - t_tr) * ((1 + out) * (1 - out)) / 2
        delta_h = np.dot(w.T, delta_o) * ((1 + hout) * (1 - hout)) / 2

        ## weight update
        dw = (dw * alpha) - np.dot(delta_o, hout.T) * (1 - alpha)
        dv = (dv * alpha) - np.dot(delta_h, p_tr.T) * (1 - alpha)
        w = w + dw * eta
        v = v + dv * eta

    if fig_err != None:
        plt.figure(fig_err), plt.plot(err_val)
    return err_val


x = np.arange(0, 2*np.pi+0.1, 0.1)
patt_tr = x.copy()
aux = np.sin(2*patt_tr) + np.random.normal(0, 0.1, patt_tr.shape)
tar_tr = 1*(aux >= 0) + (-1)*(aux < 0)
patt_val = x[5:]
aux = np.sin(2*patt_val) + np.random.normal(0, 0.1, patt_val.shape)
tar_val = 1*(aux >= 0) + (-1)*(aux < 0)

t1 = time.time()
out_val = batch_rbf(patt_tr, tar_tr, patt_val, tar_val, n_hidden = 20, width = 0.01)
print('batch was {:.4f} seconds'.format(time.time()-t1))
t2 = time.time()
out_val_1h = delta_1hlayer(patt_tr, patt_val, tar_tr, tar_val, n_hidden = 20, alpha = 0.9, fig_err = 1)
print('perceptron 1hlayer was {:.4f} seconds'.format(time.time()-t2))
plt.show()

# ERROR EN L'ERROR DEL PERCEPTRON :/
