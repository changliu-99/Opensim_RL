#test
import numpy as np
a = np.array([0,1,3])
b = np.array([0,2])

c = np.where(a[b]>1)

for i in a:
    print(i)
print(c)
