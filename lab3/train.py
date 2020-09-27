import numpy as np

def hopfield(train, recall, max_epochs = 10):

    P = recall.shape[0]
    N = recall.shape[1] # number of units

    w = 1 / N * train.T @ train
    # w = np.random.uniform(0, 1, (N, N))

    converged = False
    max_epochs = 10
    e = 0

    while not converged and e < max_epochs:

        mult = recall @ w
        pp = 1*(mult >= 0) - 1*(mult < 0)
        conv = np.sum(recall == pp, axis = 1) == N # (P,)

        if np.all(conv):
            converged = True
        else:
            recall = pp
            e += 1

    print('did' + (not converged)*' not', 'converge in', e, 'epochs')

    return recall
