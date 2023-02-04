import numpy as np

np_arr = np.array([0, 1, 2, 3])
l = []
l.append(np_arr)
l.append(np_arr)
print(len(l))
new_np_arr = np.array(l)
print(new_np_arr)
exit(0)
