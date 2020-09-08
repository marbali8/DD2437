import numpy as np
import matplotlib.pyplot as plt

from delta import delta_rule_1hlayer_batch_val
from extras import plot_error

## 3.2.3 Function approximation

x = np.arange(-5, 5+0.5, 0.5).reshape((-1, 1)) # (21, 1)
y = x.copy() # (21, 1)
z = np.dot(np.exp(-x * x * 0.1), np.exp(-y * y * 0.1).T) - 0.5 # (21, 21)
ndata = z.size
targets = np.reshape(z, (1, ndata)) # (1, ndata)
[xx, yy] = np.meshgrid(x, y)
patterns = np.concatenate([np.reshape(xx, (1, ndata)), np.reshape(yy, (1, ndata))])  # (2, ndata)
reorder_idx = np.arange(ndata)
np.random.shuffle(reorder_idx)
i_tr = reorder_idx[:int(ndata*0.75)]
i_val = reorder_idx[int(ndata*0.75):]

e_tr, _ = delta_rule_1hlayer_batch_val(patterns[:, i_tr], patterns[:, i_val], targets[:, i_tr], targets[:, i_val], n_hidden = 10, plot_d = False, plot_val = True, epochs = 200)
plt.plot(targets.T[::4])

plot_error(e_tr[::5])
plt.show()
plt.close()
