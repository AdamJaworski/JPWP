from time import time


def measure_time(func):
    def wrapper(*args):
        start = time()
        func(args[0])
        return time() - start
    return wrapper
