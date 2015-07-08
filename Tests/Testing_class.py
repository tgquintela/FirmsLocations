
import multiprocessing as mp

class Test():
#    def __init__(self, n_process):
#        self.procs = n_process

    def compute(self, data):
        pool = mp.Pool(self.procs)
        result = pool.map(f, data)
        return result


class Particular_test(Test):
    def __init__(self, n_process):
        self.procs = n_process



def f(x):
    return x

    #counts = [np.count_nonzero(np.equal(vals, v)) for v in range(n_vals)]
    #return counts
