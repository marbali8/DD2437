import numpy as np
import matplotlib.pyplot as plt

from delta import delta_rule_1hlayer_batch
from extras import plot_error

## data
patterns = np.array([[-1, 1, -1, 1], [-1, -1, 1, 1]]) # col = inputs, row = dim
targets = np.array([[-1, 1, 1, -1]]) # col = inputs, # row = outputs?
mse = delta_rule_1hlayer_batch(patterns, targets)

plot_error(mse)
