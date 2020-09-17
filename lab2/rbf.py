import numpy as np
import matplotlib.pyplot as plt

def batch_rbf(p_tr, t_tr, p_val, t_val, n_hidden = 10, mu = [], width = 0.1):

    ## initialisation
    ndata = p_tr.shape[0]

    if np.array(mu).size != n_hidden:
        mu = np.linspace(np.min(p_tr), np.max(p_tr), n_hidden)
    var = np.ones(n_hidden)*width
    w = np.random.randn(n_hidden)

    ## forward pass
    phi = np.zeros((ndata, n_hidden))
    for n in range(n_hidden):

        phi[:, n] = np.exp(-(p_tr - mu[n])**2/(2*var[n]))
        # out = w[i]*oin

    w = np.linalg.solve(np.dot(phi.T, phi), np.dot(phi.T, t_tr)) # eq (8)

    ## validation
    out_val = np.array([])
    for x in p_val:
        fhat = 0
        for n in range(n_hidden):
            fhat = fhat + w[n] * np.exp(-(x - mu[n])**2/(2*var[n]))
        out_val = np.append(out_val, fhat)

    return out_val

def delta_rbf(p_tr, t_tr, p_val, t_val, eta = 0.001, epochs = 20, n_hidden = 10, mu = [], width = 0.1, fig_err = None): # no bias

    # ASSUMES ALL HAS 1 DIMENSION (N,)

    ## initialisation
    ndata = p_tr.shape[0]

    if np.array(mu).size != n_hidden:
        mu = np.linspace(np.min(p_tr), np.max(p_tr), n_hidden)
    var = np.ones(n_hidden)*width
    w = np.random.randn(n_hidden) # from input to output
    dw = np.zeros(w.shape)
    err = np.array([])

    for ep in range(epochs): # [0, epochs-1]

        idx = np.arange(ndata)
        np.random.shuffle(idx)
        p_tr = p_tr[idx]
        t_tr = t_tr[idx]

        ## forward pass
        for i, xk in enumerate(p_tr):

            phi_xk = np.exp(-(xk - mu)**2/(2*var))
            e = t_tr[i] - np.dot(phi_xk.T, w)
            dw = dw + np.dot(e, phi_xk)

        ## weight update
        w = w + dw * eta

        ## validation
        fhat = np.array(p_val.shape)
        for n in range(n_hidden):
            phi_n = np.exp(-(p_val - mu[n])**2/(2*var[n]))
            fhat = fhat + w[n] * phi_n
        out_val = fhat.copy() - p_val.shape[0]
        err = np.append(err, np.mean(np.abs(out_val - t_val)))


    if fig_err != None:
        plt.figure(fig_err), plt.plot(err)

    return out_val
