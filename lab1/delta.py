import numpy as np

def delta_rule(patterns, targets):

    ## initialisation
    Nhidden = 2
    nIt = 60 # ?????
    stepL = 1 # ?????
    eta = 0.001 # learning rate
    alpha = 0.9 # momentum
    epochs = 1 # how many times the batch goes through the network

    ndata = patterns.shape[1] # cols

    bias = np.ones((1, ndata))
    w = np.random.randn(1, patterns.shape[0]+1)
    v = np.random.randn(1, Nhidden)
    dw = 0
    dv = 0


    for e in range(epochs): # [0, epochs-1]

        ## forward pass (layer for layer, from start to end)
        hin = np.dot(w, np.concatenate((patterns, bias))) # neuron "sum" before act, adds bias row
        hout = np.concatenate((2 / (1 + np.exp(-hin)) - 1, bias)) # activation phi
        oin = np.dot(v, hout) # (1, patterns.shape[1]+1)
        out = 2 / (1 + np.exp(-oin)) - 1 # (1, patterns.shape[1]+1)

        ## backward pass (error signal for each node, from end to start)
        delta_o = (out - targets) * ((1 + out) * (1 - out)) / 2 # dphi last, (1, patterns.shape[1]+1)
        delta_h = (v.T * delta_o) * ((1 + hout) * (1 - hout)) / 2 # dphi first
        delta_h = delta_h[0: Nhidden, :] # (Nhidden, patterns.shape[1]+1)

        ## weight update
        dw = (dw * alpha) - np.dot(delta_h, patterns.T) * (1 - alpha)
        dv = (dv * alpha) - np.dot(delta_o, hout.T) * (1 - alpha)
        w = w + dw * eta
        v = v + dv * eta


    return w, v
