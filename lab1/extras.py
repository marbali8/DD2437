import matplotlib.pyplot as plt

def plot_error(mse, title = '', new_fig = True):

    if new_fig:
        plt.figure()
    plt.plot(range(1, mse.shape[0]+1), mse)
    plt.ylabel('error'), plt.xlabel('epoch'), plt.xticks(range(1, mse.shape[0]+1))
    plt.title(title)
