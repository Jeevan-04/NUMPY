import numpy as np
import os

# Ensure the directory exists
directory = '/Users/apple/Desktop/NUMPY/17_Performance_Optimization'
os.makedirs(directory, exist_ok=True)

# Set print options
np.set_printoptions(precision=2, suppress=True)
print("Print options set:", np.get_printoptions())

# Set buffer size
np.setbufsize(8192)
print("Buffer size set to 8192 bytes")

# Set and get error handling
np.seterr(all='ignore')
print("Error handling set to ignore:", np.geterr())

# Memory-mapped file
data_path = os.path.join(directory, 'data.dat')
data = np.memmap(data_path, dtype=np.float32, mode='w+', shape=(100,))
data[:] = np.arange(100)
print("Memory-mapped data:", data[:10])

# Load with mmap_mode
npy_path = os.path.join(directory, 'data.npy')
np.save(npy_path, data)
loaded_data = np.load(npy_path, mmap_mode='r')
print("Loaded data with mmap_mode:", loaded_data[:10])

# Einstein summation
a = np.arange(25).reshape(5, 5)
b = np.arange(5)
einsum_result = np.einsum('ij,j->i', a, b)
print("Einstein summation result:", einsum_result)

# Vectorize
def myfunc(x):
    return x + 1
vectorized_func = np.vectorize(myfunc)
print("Vectorized function result:", vectorized_func([1, 2, 3]))

# Interpolation
x = np.arange(5)
xp = [0, 1, 2]
fp = [0, 1, 4]
interp_result = np.interp(x, xp, fp)
print("Interpolation result:", interp_result)

# From iterator
iterable = (x for x in range(5))
fromiter_result = np.fromiter(iterable, dtype=int)
print("From iterator result:", fromiter_result)

# Bin count
arr = np.array([0, 1, 1, 2, 2, 2, 3])
bincount_result = np.bincount(arr)
print("Bin count result:", bincount_result)

# Histogram
hist_result, bin_edges = np.histogram(arr, bins=4)
print("Histogram result:", hist_result, "Bin edges:", bin_edges)

# Matrix multiplication
matmul_result = np.matmul(a, b)
print("Matrix multiplication result:", matmul_result)

# Dot product
dot_result = np.dot(a, b)
print("Dot product result:", dot_result)

# Multi dot product
b_reshaped = b.reshape(5, 1)  # Reshape b to be 2-dimensional
multi_dot_result = np.linalg.multi_dot([a, b_reshaped, b_reshaped.T])
print("Multi dot product result:", multi_dot_result)

# Replace NaNs with numbers
nan_array = np.array([1, 2, np.nan, 4])
nan_to_num_result = np.nan_to_num(nan_array)
print("NaN to num result:", nan_to_num_result)

# Clip
clip_result = np.clip(a, 10, 20)
print("Clip result:", clip_result)

# Tile
tile_result = np.tile(b, (2, 2))
print("Tile result:", tile_result)

# Broadcast to
broadcast_result = np.broadcast_to(b, (5, 5))
print("Broadcast to result:", broadcast_result)

# Arange
arange_result = np.arange(10)
print("Arange result:", arange_result)

# Ones
ones_result = np.ones((2, 2))
print("Ones result:", ones_result)

# Zeros
zeros_result = np.zeros((2, 2))
print("Zeros result:", zeros_result)

# Empty
empty_result = np.empty((2, 2))
print("Empty result:", empty_result)

# Zeros like
zeros_like_result = np.zeros_like(a)
print("Zeros like result:", zeros_like_result)

# Ones like
ones_like_result = np.ones_like(a)
print("Ones like result:", ones_like_result)

# Full
full_result = np.full((2, 2), 7)
print("Full result:", full_result)
