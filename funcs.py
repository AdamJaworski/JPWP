import numpy as np
from numba import jit, prange
from utilities import measure_time

total_sum_global = 0


@measure_time
def square_sum_baseline(input_array: list):
    global total_sum_global
    for num in input_array:
        total_sum_global += num * num
    return total_sum_global


@measure_time
def square_sum_no_global(input_array: list):
    total_sum = 0
    for num in input_array:
        total_sum += num * num
    return total_sum


@measure_time
def square_sum_vec(input_array: np.ndarray):
    return np.sum(input_array ** 2)


@measure_time
@jit
def square_sum_jit(input_array: np.ndarray):
    total_sum = 0
    for num in input_array:
        total_sum += num * num
    return total_sum


@measure_time
@jit(parallel=True)
def square_sum_prange(input_array: np.ndarray):
    total_sum = 0
    for i in prange(len(input_array)):
        total_sum += input_array[i] * input_array[i]
    return total_sum


@measure_time
@jit(parallel=True)
def square_sum_simd_numpy(input_array: np.ndarray):
    return np.sum(np.square(input_array))
