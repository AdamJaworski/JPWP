from funcs import *

# input
n = 100_000_000
input_list = [*range(n)]
input_array = np.array(input_list)

if __name__ == "__main__":
    print(f"Base line: {square_sum_baseline(input_list):.4f}    s")
    print(f"Local variables: {square_sum_no_global(input_list):.4f}s")
    print(f"Vectorization: {square_sum_vec(input_array):.4f}s")

    init_input = np.zeros(1, input_array.dtype)
    square_sum_jit(init_input)
    square_sum_prange(init_input)
    square_sum_simd_numpy(init_input)

    print(f"JIT: {square_sum_jit(input_array):.4f}s")
    print(f"JIT + SIMD: {square_sum_prange(input_array):.4f}s")
    print(f"JIT + NumPy: {square_sum_simd_numpy(input_array):.4f}s")


