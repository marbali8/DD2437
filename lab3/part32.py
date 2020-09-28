import numpy as np
import matplotlib.pyplot as plt

from train import hopfield

P = 11
N = 1024 # number of units
data = np.loadtxt('pict.dat', delimiter = ',')

input = np.zeros((P, N))

for i, begin in enumerate(np.arange(0, data.shape[0], N)):

    input[i, :] = data[begin:begin+N]

# question 1: first 3 patterns stable

# train = input[:3, :]
# test = input[:3, :]
#
# output = hopfield(train, test, seq = True)
# correct = np.sum(output == test, axis = 1) == N
#
# print(np.where(correct)[0])

# e = 0, all 3 patterns are attractors obviously

# question 2: p10

# train = input[:3, :]
# test = input[9, :].reshape((1, -1)) # p10
#
# output = hopfield(train, test, seq = False)
# plt.figure(), plt.imshow(input[0, :].reshape((32, 32))) # p1
# plt.figure(), plt.imshow(test.reshape((32, 32)))
# plt.figure(), plt.imshow(output.reshape((32, 32)))
#
# plt.show()

# converges to correct attractor

# question 2: p11

# train = input[:3, :]
# test = input[10, :].reshape((1, -1)) # p11
#
# output = hopfield(train, test)
# plt.figure(), plt.imshow(input[1, :].reshape((32, 32))) # p2
# plt.figure(), plt.imshow(input[2, :].reshape((32, 32))) # p3
# plt.figure(), plt.imshow(test.reshape((32, 32)))
# plt.figure(), plt.imshow(output.reshape((32, 32)))
#
# plt.show()

# doesn't converge to correct attractor

# question 3

train = input[[5, 8, 1], :]
test = input[9, :].reshape((1, -1)) # p10

output = hopfield(train, test, seq = True)
# plt.figure(), plt.imshow(train[0, :].reshape((32, 32))) # p5
# plt.figure(), plt.imshow(train[1, :].reshape((32, 32))) # p8
# plt.figure(), plt.imshow(train[2, :].reshape((32, 32))) # p1
# plt.figure(), plt.imshow(test.reshape((32, 32)))
# plt.figure(), plt.imshow(output.reshape((32, 32)))

plt.show()

# seq is not working... it's like it never changes! idk why
