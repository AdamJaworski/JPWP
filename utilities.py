from time import time


def measure_time(func):
    def wrapper(*args):
        start = time()
        func(args[0])
        total_time = (time() - start)
        #print(f"Execution time: {total_time:.4f}s\n")
        return total_time
    return wrapper
