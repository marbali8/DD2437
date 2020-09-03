import numpy as np
from delta import delta_rule

## create data
n = 100;
mA = [[1.0], [0.5]]
sigmaA = 0.5
mB = [[-1.0], [0.0]]
sigmaB = 0.5

classA = np.random.randn(2, n) * sigmaA + mA # (2, n)
classA = np.concatenate([classA, np.zeros((1, n))])  # (3, n)
classB = np.random.randn(2, n) * sigmaB + mB # (2, n)
classB = np.concatenate([classB, np.ones((1, n))])  # (3, n)

data = np.concatenate([classA, classB], axis = 1) # (3, 2n)
np.random.shuffle(data.T) # (3, 2n)

patterns = data[:2, :] # (2, 2n)
target = data[-1:, :] # (1, 2n)

w, v = delta_rule(patterns, targets)
