import numpy as np
from delta import delta_rule

def prelab():

    ## data
    patterns = np.array([[-1, 1, -1, 1], [-1, -1, 1, 1]]) # col = inputs, row = dim
    targets = np.array([-1, 1, 1, -1]) # col = inputs, # row = outputs?

    w, v = delta_rule()
