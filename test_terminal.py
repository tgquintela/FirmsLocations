


#### Test about counting in a array
import time
import numpy as np

## Messages
mess = "Total time %f seconds for %s. Ratio %f per size."
## input values
n = 100000
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



