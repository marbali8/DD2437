import numpy as np
import matplotlib.pyplot as plt

def perceptron_rule_0hlayer_batch(patterns, targets, eta = 0.001, alpha = 0.9, epochs = 20, print_acc = False):

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
    mse = np.array([])
    acc = np.array([])

    x1 = np.arange(-3, 3, 0.5)

    for e in range(epochs): # [0, epochs-1]

        ## forward pass
        oin = np.dot(w, np.concatenate((patterns, bias))) # neuron "sum" before act, adds bias row
        out = -1*(oin <= 0) + 1*(oin > 0) # activation (tlu)

        mse = np.append(mse, ((targets - out)**2).mean())
        acc = np.append(acc, (np.all(out == targets, axis = 0)).mean())

        ## backward pass (error signal for each node, from end to start)

        ## weight update
        dw = np.dot(targets - out, np.concatenate((patterns, bias)).T) # (targets.shape[0], in_dim+1)
        w = w + dw * eta

    if print_acc:
        print('{:.4f}'.format(acc[-1]))
    decision = - 1 / w[0, 1] * (w[0, 2] + w[0, 0] * x1) # from Wx = 0 (indexes are translated 1 bc w0 in formula is w2 here)
    plt.plot(x1, decision, c = 'yellow')
    return mse

def perceptron_rule_0hlayer_batch_nbias(patterns, targets, eta = 0.001, alpha = 0.9, epochs = 20):

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
    mse = np.array([])

    x1 = np.arange(-3, 3, 0.5)

    for e in range(epochs): # [0, epochs-1]

        ## forward pass
        oin = np.dot(w, patterns) # neuron "sum" before act, adds bias row
        out = -1*(oin <= 0) + 1*(oin > 0) # activation (tlu)

        mse = np.append(mse, ((out - targets)**2).mean())

        ## backward pass (error signal for each node, from end to start)

        ## weight update
        dw = np.dot(targets - out, patterns.T) # (targets.shape[0], in_dim)
        w = w + dw * eta

    decision = - 1 / w[0, 1] * w[0, 0] * x1
    plt.plot(x1, decision, c = 'yellow')
    return mse
