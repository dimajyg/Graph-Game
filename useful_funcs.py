import numba as nb
import numpy as np
from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

@nb.njit
def filter_equal(arr,value : int | float):
    j = 0
    for i in range(arr.size):
        if arr[i] != value:
            j += 1
    result = np.empty(j, dtype=arr.dtype)
    j = 0
    for i in range(arr.size):
        if arr[i] != value:
            result[j] = arr[i]
            j += 1
    return result