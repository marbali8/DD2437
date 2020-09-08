import matplotlib.pyplot as plt

def plot_error(e, title = '', new_fig = True):

    if new_fig:
        plt.figure()
    plt.plot(range(1, e.shape[0]+1), e)
    plt.ylabel('error'), plt.xlabel('epoch'), plt.xticks(range(1, e.shape[0]+1))
    plt.title(title + ', final error: ' + str(e[-1]))
