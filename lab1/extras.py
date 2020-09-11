import matplotlib.pyplot as plt
import numpy as np

def plot_error(e, title = '', new_fig = True):

    if new_fig:
        plt.figure()
    plt.plot(range(1, e.shape[0]+1), e)
    plt.ylabel('error'), plt.xlabel('epoch'), #plt.xticks(range(1, e.shape[0]+1))
    plt.title(title)

def compute_boundary(input, v, w, xrange = None, yrange = None):

    ndata = input.shape[1] # cols
    in_dim = input.shape[0]

    if in_dim != 2:
        return

    if xrange == None:
        xx = np.linspace(np.min(input[0]), np.max(input[0]), num = 100) # (100,)
    else:
        xx = np.linspace(xrange[0], xrange[1], num = 100) # (100,)
    if yrange == None:
        yy = np.linspace(np.min(input[1]), np.max(input[1]), num = 100) # (100,)
    else:
        yy = np.linspace(yrange[0], yrange[1], num = 100) # (100,)

    boundary = np.array([])

    for x in xx:
        for i, y in enumerate(yy):

            bias = np.ones((1, 1))
            data = np.array([x, y]).reshape((in_dim, 1)) # (2, 1)

            ## forward pass (layer for layer, from start to end)
            hin = np.dot(v, np.concatenate((data, bias))) # neuron "sum" before act, adds bias row
            hout = np.concatenate((2 / (1 + np.exp(-hin)) - 1, bias)) # activation phi
            oin = np.dot(w, hout)

            if i == 0:
                out = 2 / (1 + np.exp(-oin)) - 1

            if out * (2 / (1 + np.exp(-oin)) - 1) <= 0:

                if boundary.size == 0:
                    boundary = data.copy()
                else:
                    boundary = np.concatenate((boundary, data), axis = 1)
                break
    return boundary
