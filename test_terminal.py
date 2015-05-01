


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



