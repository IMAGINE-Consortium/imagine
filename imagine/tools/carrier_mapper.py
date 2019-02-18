import numpy as np

def infinity_mapper(x, a=-np.inf, m=0, b=np.inf):
    '''
    Maps x from [-inf, inf] into the interval [a, b], where x=0 -> m
    '''
    if a == -np.inf and b == np.inf and m == 0:
        return x
    x = np.float(x)
    a = np.float(a)
    b = np.float(b)
    if m is None:
        if a == -np.inf and b == np.inf:
            m = 0
        else:
            m = a + (b-a)/2.
    else:
        m = np.float(m)
    # map x from [-inf, inf] to [0, 1]
    y = np.arctan(x)/np.pi + 0.5
    # compute where m would lie in [0, 1]
    n = (m - a)/(b - a)
    # strech y, such that x=0 -> n
    y = y**np.emath.logn(0.5, n)
    # strech y to the interval [a,b]
    y = y*(b-a) + a
    return y

def unity_mapper(x, a=0, b=1):
    '''
    Maps x from [0, 1] into the interval [a, b]
    '''
    return float(x) * (float(b)-float(a)) + float(a)
