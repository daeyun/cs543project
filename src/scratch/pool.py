from multiprocessing import Pool
import time
import itertools

"""
playing with multiprocessing.Pool
"""

a = 0

def func(num):
    print "func:" + str(num)

def worker(num):
    print [num]*num*num
    return [num]*num*num

pool = Pool(processes=5)
result = pool.map(worker, [1,2,3,4,5])
pool.close()
pool.join()
print result
print list(itertools.chain.from_iterable(result))
print "exiting"