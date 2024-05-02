import numpy as np


array = [1, np.array([2]), np.array([3]), 4, 5, 6]

for i in range(1,3):
	array[i] = array[i][0]
print(array)

print(np.cumsum(array))

