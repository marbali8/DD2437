import numpy as np
import matplotlib.pyplot as plt

def batch_rbf(p_tr, t_tr, p_val, t_val, n_hidden, width, mu = []):

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

def delta_rbf(p_tr, t_tr, p_val, t_val, eta = 0.001, epochs = 20, n_hidden = 10, mu = [], width = 0.1, fig_err = None, fig_mu = None): # no bias

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
        p_tr_s = p_tr[idx]
        t_tr_s = t_tr[idx]

        ## forward pass
        for i, xk in enumerate(p_tr_s):

            phi_xk = np.exp(-(xk - mu)**2/(2*var))
            e = t_tr_s[i] - np.dot(phi_xk.T, w)

            ## weight update
            dw = e * phi_xk
            w = w + dw * eta

        ## validation
        out_val = np.empty(p_val.shape)
        out_tr = np.empty(p_tr.shape)

        for i, xk in enumerate(p_val):

            phi_xk = np.exp(-(xk - mu)**2/(2*var))
            out_val[i] = np.dot(phi_xk.T, w)

        for i, xk in enumerate(p_tr):

            phi_xk = np.exp(-(xk - mu)**2/(2*var))
            out_tr[i] = np.dot(phi_xk.T, w)

        err = np.append(err, np.sqrt(np.mean((out_val - t_val)**2)))

    if fig_err != None:
        plt.figure(fig_err), plt.plot(err)
        print('delta: {:.4f}'.format(err[-1]))

    if fig_mu != None:
        plt.figure(fig_mu)
        plt.xlim(-1, np.max(p_tr) + 1)
        y = np.ones(n_hidden)
        plt.plot(mu, y*np.random.randint(5), 'o', ms = 5)
        plt.axis('off')

    return out_tr


def delta_rbf_cl(p_tr, t_tr, p_val, t_val, eta = 0.001, epochs = 20, n_hidden = 10, mu = [], width = 0.1, nhood = 1, fig_err = None, fig_mu = None): # no bias

    # ASSUMES ALL HAS 1 DIMENSION (N,)

    ## initialisation
    ndata = p_tr.shape[0]

    # mu are the weights from input to rbfs (see lecture 6)
    if np.array(mu).size != n_hidden:
        mu = np.linspace(np.min(p_tr), np.max(p_tr), n_hidden)
    var = np.ones(n_hidden)*width
    w = np.random.randn(n_hidden) # from rbfs to output
    dw = np.zeros(w.shape)
    err = np.array([])

    for ep in range(epochs): # [0, epochs-1]

        idx = np.arange(ndata)
        np.random.shuffle(idx)
        p_tr_s = p_tr[idx]
        t_tr_s = t_tr[idx]

        # find closest rbf to random datapoint
        rand_idx = idx[0]
        phi_xk = np.exp(-(p_tr_s[rand_idx] - mu)**2/(2*var))
        e = t_tr_s[rand_idx] - phi_xk * w
        min_n = np.argsort(np.abs(e))
        for i in range(nhood):
            d = t_tr_s[rand_idx] - mu[min_n[i]]
            mu[min_n[i]] += (e[min_n[i]] > 0)*eta*d/(i+1)

        # # assign new widths (var) for the rbfs
        # for n in range(n_hidden):
        #     dist = np.abs(t_tr_s - mu[n]) # abs bc it's 1d!
        #     var[n] = np.sqrt(np.sort(dist)[int(ndata/n_hidden)])

        ## forward pass
        for i, xk in enumerate(p_tr_s):

            phi_xk = np.exp(-(xk - mu)**2/(2*var))
            e = t_tr_s[i] - np.dot(phi_xk.T, w)

            ## weight update
            dw = e * phi_xk
            w = w + dw * eta

        ## validation
        out_val = np.empty(p_val.shape)
        out_tr = np.empty(p_tr.shape)

        for i, xk in enumerate(p_val):

            phi_xk = np.exp(-(xk - mu)**2/(2*var))
            out_val[i] = np.dot(phi_xk.T, w)

        for i, xk in enumerate(p_tr):

            phi_xk = np.exp(-(xk - mu)**2/(2*var))
            out_tr[i] = np.dot(phi_xk.T, w)

        err = np.append(err, np.sqrt(np.mean((out_val - t_val)**2)))

    if fig_err != None:
        plt.figure(fig_err), plt.plot(err)
        print('compe: {:.4f}'.format(err[-1]))

    if fig_mu != None:
        plt.figure(fig_mu)
        plt.xlim(-1, np.max(p_tr) + 1)
        y = np.ones(n_hidden)
        plt.plot(mu, y*np.random.randint(5), 'o', ms = 5)
        plt.axis('off')

    return out_tr
