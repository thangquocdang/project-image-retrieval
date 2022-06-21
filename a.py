import numpy as np
from numpy.core.fromnumeric import mean

l = np.array([7,15,10,12,8,6,15])
y = np.array([18,6,16,10,11,12,17])
x = y**2
print(np.sum(x))
print(np.sum(y))
print(np.sum(l*y))