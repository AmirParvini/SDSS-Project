import numpy as np
import random
from scipy.spatial.distance import cdist

a = [[1,2,3],[4,1,6]]
print(np.min(a, axis=0))