from typing import Callable
from time import time


def time_count(func: Callable):
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        print(time() - start)
        return result
    return wrapper
