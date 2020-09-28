import numpy as np
import matplotlib.pyplot as plt

def hopfield(train, recall, seq = False, max_epochs = 1000):

    P = recall.shape[0]
    N = recall.shape[1] # number of units

    w = 1 / N * train.T @ train
    # w = np.random.uniform(0, 1, (N, N))

    converged = False
    e = 0
    _recall = recall.copy()

    while not converged and e < max_epochs:

        if not seq:
            mult = _recall @ w
            pp = 1*(mult >= 0) - 1*(mult < 0)
            conv = np.sum(_recall == pp, axis = 1) == N # (P,)

            if np.all(conv):

                converged = True
            else:

                _recall = pp
        else:
            for it in range(10000):

                aux = _recall.copy()
                i = np.random.randint(0, N)
                # i = it

                mult = _recall @ w[:, i]

                # _recall[:, i] = 1*(mult >= 0) - 1*(mult < 0)
                _recall[:, i] = np.sign(mult)
                if np.all(aux != np.sign(mult)):
                    print(it, 'change')

                if e == 0 and it % 1000 == 0 and _recall.size == N:

                    mult = _recall @ w
                    pp = 1*(mult >= 0) - 1*(mult < 0)

                    plt.figure(), plt.imshow(pp.reshape((32, 32))), plt.show()

            mult = _recall @ w
            pp = 1*(mult >= 0) - 1*(mult < 0)
            conv = np.sum(_recall == pp, axis = 1) == N # (P,)

            if np.all(conv):

                converged = True

        e += 1

    print('did' + (not converged)*' not', 'converge in', e, 'epoch' + (e != 1)*'s')

    return _recall
