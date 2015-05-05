


#### Test about counting in a array
import time
import numpy as np

## Messages
mess = "Total time %f seconds for %s. Ratio %f per size."
## input values
n = 1000
m = 50

a = np.random.randint(0,m,n)

t0 = time.time()
count = np.zeros(m)
for i in range(m):
    count[i] = np.sum(a == i)
t1 = time.time()-t0
print (mess % (t1, str(n), t1/n))


t0 = time.time()
count = np.zeros(m)
for i in range(m):
    count[i] = np.count_nonzero(a == i)
t1 = time.time()-t0
print (mess % (t1, str(n), t1/n))

t0 = time.time()
count = np.array([np.count_nonzero(a == i) for i in range(m)])
t1 = time.time()-t0
print (mess % (t1, str(n), t1/n))

t0 = time.time()
count = np.array([np.count_nonzero(np.equal(a, i)) for i in range(m)])
t1 = time.time()-t0
print (mess % (t1, str(n), t1/n))


###############################################################################################
### count in an array


import time
import pandas as pd
import os
from os.path import join
folder = 'Data/Outputs/neighs/neighs_2_0'

#t0 = time.time()
#data = pd.read_csv('/home/tono/mscthesis/code/Data/Outputs/neighs/neighs_0_5/neighs_0001', sep=';', index_col=0)
#n = data.shape[0]
#t = time.time()-t0
#print m % (t, str(n), t/n)

t0 = time.time()
n = 0
files = os.listdir(folder)
for f in files:
    data = pd.read_csv(join(folder, f), sep=';', index_col=0)
    for j in range(data.shape[0]):
        neighs = data.loc[data.index[0],'neighs'].split(',')
        neighs = [int(e) for e in neighs]
    n += data.shape[0]
t = time.time()-t0
print m % (t, str(n), t/n)


### Retrieve from numpy vs pandas
n, m, s = 1000000,10, 1000
a = np.random.random((n, m))
df = pd.DataFrame(a)
idxs = np.random.randint(0, n, s)

t0 = time.time()
val1 = df.loc[idxs, :]
print "Time expended is %f seconds." % (time.time()-t0)
t0 = time.time()
val1 = a[idxs, :]
print "Time expended is %f seconds." % (time.time()-t0)



### Counting
import numpy as np
import time
from collections import Counter

def f1(a, m):
    count = np.zeros(m)
    c = Counter(a)
    count[c.keys()] = c.values()
    return c
def f2(a, m):
    return np.array([np.count_nonzero(np.equal(a, v)) for v in range(m)])
def f3(a, m):
    count = np.zeros(m)
    for i in xrange(a.shape[0]):
        count[a[i]] += 1
    return count

mess = "Time expended in method %s is %f seconds."
#### parameters
m = 100
n = 100000
### Needed vars
a = np.random.randint(0, m, n)


t0 = time.time()
res1 = f1(a, m)
print mess % ("f1", time.time()-t0)
t0 = time.time()
res2 = f2(a, m)
print mess % ("f2", time.time()-t0)
t0 = time.time()
res3 = f3(a, m)
print mess % ("f3", time.time()-t0)


### Retrieve numpy with list and with array
import numpy as np
import time


def f1(a, idxs):
    idxs = np.array(idxs)
    return a[idxs, 10]
def f2(a, idxs):
    return a[idxs, 10]

mess = "Time expended in method %s is %f seconds."
#### parameters
m = 100
n = 100000
n1 = 100000
### Needed vars
a = np.random.randint(0, m, (n, 60))
idxs = np.random.randint(0, m, n1)


t0 = time.time()
res1 = f1(a, idxs)
print mess % ("f1", time.time()-t0)
t0 = time.time()
res2 = f2(a, idxs)
print mess % ("f2", time.time()-t0)

### Measure distances in different methods
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

#data = pd.read_csv(, sep=';')
#data = np.array(data[['ES-X', 'ES-Y']])
data = np.array([[41.407579, 2.209777], [41.380601, 2.122963]])
res = [7.84]
data2 = np.array([[41.380601, 2.122963], [41.300606, 2.116594]])
res2 = [8,89]
data3 = np.array([[41.407579, 2.209777], [41.380601, 2.122963], [41.300606, 2.116594]])

def f1(locs):
    locs = np.pi/180.*locs
    locs[:, 1] = locs[:, 1]*np.cos(locs[:, 0])
    dist = 6371.009 * pdist(locs)
    return dist
    


