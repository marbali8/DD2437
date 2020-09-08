import numpy as np
import matplotlib.pyplot as plt

def delta_rule_1hlayer_batch(patterns, targets, n_hidden = 2, eta = 0.001, alpha = 0.9, epochs = 20, plot = True):

    ## initialisation
    # n_hidden = 2
    # nIt = 60 # ?????
    # stepL = 1 # ?????
    # eta = 0.001 # learning rate
    # alpha = 0.9 # momentum
    # epochs = 20 # how many times the batch goes through the network

    ndata = patterns.shape[1] # cols
    in_dim = patterns.shape[0]

    bias = np.ones((1, ndata))
    v = np.random.randn(n_hidden, in_dim+1) # from input to hidden
    w = np.random.randn(targets.shape[0], n_hidden+1) # from hidden to output
    dw = np.zeros(w.shape)
    dv = np.zeros(v.shape)
    err = np.array([])

    x1 = np.arange(-3, 3, 0.5)

    for e in range(epochs): # [0, epochs-1]

        ## forward pass (layer for la2yer, from start to end)
        hin = np.dot(v, np.concatenate((patterns, bias))) # neuron "sum" before act, adds bias row 3,8
        hout = np.concatenate((2 / (1 + np.exp(-hin)) - 1, bias)) # activation phi 4, 8
        oin = np.dot(w, hout) # (1, patterns.shape[1]+1) 8,8
        out = 2 / (1 + np.exp(-oin)) - 1 # (1, ndata) = targets.shape 8,8

        err = np.append(err, ((out - targets)**2).mean())

        ## backward pass (error signal for each node, from end to start)
        delta_o = (out - targets) * ((1 + out) * (1 - out)) / 2 # dphi last, (1, ndata) 8,8
        delta_h = np.dot(w.T, delta_o) * ((1 + hout) * (1 - hout)) / 2 # dphi first 4,8 8,8
        delta_h = delta_h[0: n_hidden, :] # (n_hidden, ndata)

        ## weight update
        dw = (dw * alpha) - np.dot(delta_o, hout.T) * (1 - alpha) # (targets.shape[0], n_hidden+1)
        dv = (dv * alpha) - np.dot(delta_h, np.concatenate((patterns, bias)).T) * (1 - alpha) # (n_hidden, in_dim+1)
        w = w + dw * eta
        v = v + dv * eta
    if plot:
        decision = - 1 / w[0, 1] * (w[0, 2] + w[0, 0] * x1) # from Wx = 0 (indexes are translated 1 bc w0 in formula is w2 here)
        plt.plot(x1, decision, c = 'pink')
    return err

def delta_rule_1hlayer_batch_val(patterns_tr, patterns_val, targets_tr, targets_val, n_hidden = 2, eta = 0.001, alpha = 0.9, epochs = 20, plot_d = True, plot_val = False):

    ## initialisation
    # n_hidden = 2
    # nIt = 60 # ?????
    # stepL = 1 # ?????
    # eta = 0.001 # learning rate
    # alpha = 0.9 # momentum
    # epochs = 20 # how many times the batch goes through the network

    ndata = patterns_tr.shape[1] # cols
    in_dim = patterns_tr.shape[0]

    bias = np.ones((1, ndata))
    bias_v = np.ones((1, patterns_val.shape[1]))
    v = np.random.randn(n_hidden, in_dim+1) # from input to hidden
    w = np.random.randn(targets_tr.shape[0], n_hidden+1) # from hidden to output
    dw = np.zeros(w.shape)
    dv = np.zeros(v.shape)
    err_tr = np.array([])
    err_val = np.array([])

    x1 = np.arange(-3, 3, 0.5)

    for e in range(epochs): # [0, epochs-1]

        ## forward pass (layer for layer, from start to end)
        hin = np.dot(v, np.concatenate((patterns_tr, bias))) # neuron "sum" before act, adds bias row
        hout = np.concatenate((2 / (1 + np.exp(-hin)) - 1, bias)) # activation phi
        oin = np.dot(w, hout) # (1, patterns.shape[1]+1)
        out = 2 / (1 + np.exp(-oin)) - 1 # (1, ndata) = targets.shape

        err_tr = np.append(err_tr, ((out - targets_tr)**2).mean())

        ## validation
        hin_val = np.dot(v, np.concatenate((patterns_val, bias_v))) # neuron "sum" before act, adds bias row
        hout_val = np.concatenate((2 / (1 + np.exp(-hin_val)) - 1, bias_v)) # activation phi
        oin_val = np.dot(w, hout_val) # (1, patterns_val.shape[1]+1)
        out_val = 2 / (1 + np.exp(-oin_val)) - 1 # (1, ndata) = targets_val.shape

        err_val = np.append(err_val, ((out_val - targets_val)**2).mean())

        ## backward pass (error signal for each node, from end to start)
        delta_o = (out - targets_tr) * ((1 + out) * (1 - out)) / 2 # dphi last, (1, ndata)
        delta_h = np.dot(w.T, delta_o) * ((1 + hout) * (1 - hout)) / 2 # dphi first
        delta_h = delta_h[0: n_hidden, :] # (n_hidden, ndata)

        ## weight update
        dw = (dw * alpha) - np.dot(delta_o, hout.T) * (1 - alpha) # (targets_tr.shape[0], n_hidden+1)
        dv = (dv * alpha) - np.dot(delta_h, np.concatenate((patterns_tr, bias)).T) * (1 - alpha) # (n_hidden, in_dim+1)
        w = w + dw * eta
        v = v + dv * eta

        if plot_val:
            if e == 0:
                plt.figure()
            if e % int(epochs/5) == 0:
                plt.plot(out_val.T, linewidth = 0.5)
            if e == epochs-1:
                plt.title('validation shapes')
                plt.legend(np.arange(int(epochs/int(epochs/5))))

    if plot_d:
        decision = - 1 / w[0, 1] * (w[0, 2] + w[0, 0] * x1) # from Wx = 0 (indexes are translated 1 bc w0 in formula is w2 here)
        plt.plot(x1, decision, c = 'pink')
    return err_tr, err_val

def delta_rule_0hlayer_batch(patterns, targets, eta = 0.001, alpha = 0.9, epochs = 20):

    ## initialisation
    # nIt = 60 # ?????
    # stepL = 1 # ?????
    # eta = 0.001 # learning rate
    # alpha = 0.9 # momentum
    # epochs = 20 # how many times the batch goes through the network

    ndata = patterns.shape[1] # cols
    in_dim = patterns.shape[0]

    bias = np.ones((1, ndata))
    w = np.random.randn(targets.shape[0], in_dim+1) # from input to output
    dw = np.zeros(w.shape)
    err = np.array([])

    x1 = np.arange(-3, 3, 0.5)

    for e in range(epochs): # [0, epochs-1]

        ## forward pass
        oin = np.dot(w, np.concatenate((patterns, bias))) # neuron "sum" before act, adds bias row
        out = -1*(oin <= 0) + 1*(oin > 0) # activation (tlu)

        err = np.append(err, ((targets - oin)**2).mean())

        ## backward pass (error signal for each node, from end to start)

        ## weight update
        dw = np.dot(targets - oin, np.concatenate((patterns, bias)).T) # (targets.shape[0], in_dim+1)
        w = w + dw * eta

    decision = - 1 / w[0, 1] * (w[0, 2] + w[0, 0] * x1) # from Wx = 0 (indexes are translated 1 bc w0 in formula is w2 here)
    plt.plot(x1, decision, c = 'pink')
    return err

def delta_rule_0hlayer_seq(patterns, targets, eta = 0.001, alpha = 0.9, epochs = 20):

    ## initialisation
    # nIt = 60 # ?????
    # stepL = 1 # ?????
    # eta = 0.001 # learning rate
    # alpha = 0.9 # momentum
    # epochs = 20 # how many times the batch goes through the network

    ndata = patterns.shape[1] # cols
    in_dim = patterns.shape[0]

    bias = np.ones((1, 1))
    w = np.random.randn(targets.shape[0], in_dim+1) # from input to output
    dw = np.zeros(w.shape)
    err = np.array([])

    x1 = np.arange(-3, 3, 0.5)

    for e in range(epochs): # [0, epochs-1]

        for i, p in enumerate(patterns.T): # (in_dim, 1)

            ## forward pass
            p = p.reshape((in_dim, 1));
            oin = np.dot(w, np.concatenate((p, bias))) # neuron "sum" before act, adds bias row
            out = -1*(oin <= 0) + 1*(oin > 0) # activation (tlu)

            err = np.append(err, ((targets[0, i] - oin)**2).mean()).reshape((-1, 1))

            ## backward pass (error signal for each node, from end to start)

            ## weight update
            dw = np.dot(targets[0, i] - oin, np.concatenate((p, bias)).T)
            w = w + dw * eta

    decision = - 1 / w[0, 1] * (w[0, 2] + w[0, 0] * x1) # from Wx = 0 (indexes are translated 1 bc w0 in formula is w2 here)
    plt.plot(x1, decision, c = 'magenta')
    return err

def delta_rule_0hlayer_batch_nbias(patterns, targets, eta = 0.001, alpha = 0.9, epochs = 20):

    ## initialisation
    # nIt = 60 # ?????
    # stepL = 1 # ?????
    # eta = 0.001 # learning rate
    # alpha = 0.9 # momentum
    # epochs = 20 # how many times the batch goes through the network

    ndata = patterns.shape[1] # cols
    in_dim = patterns.shape[0]

    w = np.random.randn(targets.shape[0], in_dim) # from input to output
    dw = np.zeros(w.shape)
    err = np.array([])

    x1 = np.arange(-3, 3, 0.5)

    for e in range(epochs): # [0, epochs-1]

        ## forward pass
        oin = np.dot(w, patterns) # neuron "sum" before act, adds bias row
        out = -1*(oin <= 0) + 1*(oin > 0) # activation (tlu)

        err = np.append(err, ((targets - oin)**2).mean())

        ## backward pass (error signal for each node, from end to start)

        ## weight update
        dw = np.dot(targets - oin, patterns.T) # (targets.shape[0], in_dim)
        w = w + dw * eta

    decision = - 1 / w[0, 1] * w[0, 0] * x1
    plt.plot(x1, decision, c = 'magenta')
    return err
