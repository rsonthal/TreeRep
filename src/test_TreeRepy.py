## Testing for TreeRepy

import numpy as np
import time
t = time.time()
import TreeRepy
print("Time for importing TreeRepy:",time.time() - t)

d = np.array([[0,2,3],[2,0,2],[3,2,0]])
t = time.time()
W = TreeRepy.TreeRep(d)
print("Time for running TreeRepy for 3X3 adjacency matrix:",time.time() - t)
print(W)

t = time.time()
W = TreeRepy.TreeRep(d)
print("Time for running TreeRepy for 3X3 adjacency matrix second time:",time.time() - t)
print(W)