from util import *
from rbm import RestrictedBoltzmannMachine
from dbn import DeepBeliefNet
import os

if __name__ == "__main__":

    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    ''' restricted boltzmann machine '''

    print ("\nStarting a Restricted Boltzmann Machine..")

    # things i could add:
    # - avoid overfitting (6.1 and 16.1)
    # - adaptive lr (7)
    # - momentum (9.1)
    # - weight decay (10.1)

    n_samples = train_imgs.shape[0]

    # 4.1 epochs

    # epochs = 20
    # batch_size = 20
    # iterations = int(n_samples / batch_size) * epochs
    #
    # rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
    #                                  ndim_hidden=500,
    #                                  is_bottom=True,
    #                                  image_size=image_size,
    #                                  is_top=False,
    #                                  n_labels=10,
    #                                  batch_size=batch_size
    # )
    #
    # _ = rbm.cd1(visible_trainset = train_imgs, n_iterations = iterations)


    # 4.1 n_hidden

    epochs = 20
    batch_size = 10
    iterations = int(n_samples / batch_size) * epochs

    loss = []
    hidden = np.arange(500, 200-100, -100)
    for n_hid in hidden:

        rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                         ndim_hidden=n_hid,
                                         is_bottom=True,
                                         image_size=image_size,
                                         is_top=False,
                                         n_labels=10,
                                         batch_size=batch_size
        )

        recon_loss = rbm.cd1(visible_trainset = train_imgs, n_iterations = iterations)
        loss.append(recon_loss)

    plt.figure()
    for i in loss:

        it = [i for (i, l) in i]
        loss = [l for (i, l) in i]
        plt.plot(it, loss)

    plt.figure(1)
    plt.legend(['Nh = ' + str(i) for i in hidden])
    plt.title('Reconstruction loss (batch 10, epochs 20)')
    plt.xlabel('iteration')
    plt.show()

    # ''' deep- belief net '''
    #
    # print ("\nStarting a Deep Belief Net..")
    #
    # dbn = DeepBeliefNet(sizes={"vis":image_size[0]*image_size[1], "hid":500, "pen":500, "top":2000, "lbl":10},
    #                     image_size=image_size,
    #                     n_labels=10,
    #                     batch_size=10
    # )
    #
    # ''' greedy layer-wise training '''
    #
    # a = input('want to clear trained rbm models? [y, n] ')
    # if a == 'y':
    #     files = os.listdir('trained_rbm')
    #     for f in files:
    #         os.remove('trained_rbm/' + f)
    #
    # dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10000)
    #
    # dbn.recognize(train_imgs, train_lbls)
    #
    # dbn.recognize(test_imgs, test_lbls)
    #
    # for digit in range(10):
    #     digit_1hot = np.zeros(shape=(1,10))
    #     digit_1hot[0,digit] = 1
    #     dbn.generate(digit_1hot, name="rbms")
    #
    # ''' fine-tune wake-sleep training '''
    #
    # a = input('want to clear trained dbn models? [y, n] ')
    # if a == 'y':
    #     files = os.listdir('trained_dbn')
    #     for f in files:
    #         os.remove('trained_dbn/' + f)
    #
    # dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10000)
    #
    # dbn.recognize(train_imgs, train_lbls)
    #
    # dbn.recognize(test_imgs, test_lbls)
    #
    # for digit in range(10):
    #     digit_1hot = np.zeros(shape=(1,10))
    #     digit_1hot[0,digit] = 1
    #     dbn.generate(digit_1hot, name="dbn")
