import multiprocessing
from functools import partial
from contextlib import contextmanager
import math
import numpy as np 

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()



def func(a):
    x=a+2
    # y = x^4
    return x

def p_f(a):
    cpus = multiprocessing.cpu_count()
    print("number_of cpus = ",cpus)

    with poolcontext(processes=cpus) as pool:
        r = pool.map(func,a)
    return r

a = np.arange(0,1000,1)

r1 = p_f(a)
# r2 = p_f2(a)
# if r1==r2:
    # print("yes")
print(r1)
# print(r2)
# print(np.isclose(r1,r2))
