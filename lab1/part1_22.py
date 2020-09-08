import numpy as np
import matplotlib.pyplot as plt

from delta import delta_rule_1hlayer_batch
from extras import plot_error

## 3.2.2 The encoder problem

patterns = -np.ones((8, 8)) + 2*np.identity(8) # (8, ndata) = (8, 8)
np.random.shuffle(patterns)
targets = patterns.copy()
e = delta_rule_1hlayer_batch(patterns, targets, n_hidden = 3, epochs = 100, plot = False)

plot_error(e, '838 hour-glass')
# plt.show()
# plt.close()


e_2 = delta_rule_1hlayer_batch(patterns, targets, n_hidden = 2, epochs = 100, plot = False)
plot_error(e_2, '828 hour-glass', new_fig = False)
plt.show()
plt.close()
