import numpy as np

print("Script is running...")  # Add this line to check if the script starts

# Reshape an array to a new shape
a = np.array([[1, 2, 3], [4, 5, 6]])
reshaped = np.reshape(a, (3, 2))
print("Reshaped array:\n", reshaped)

# Resize an array, possibly repeating elements to fit the new size
resized = np.resize(a, (3, 3))
print("Resized array:\n", resized)

# Flatten a multi-dimensional array into a 1D array
raveled = np.ravel(a)
print("Raveled array:\n", raveled)

# Return a flattened 1D array
flattened = a.flatten()
print("Flattened array:\n", flattened)

# Transpose the dimensions of an array
transposed = np.transpose(a)
print("Transposed array:\n", transposed)

# Swap two axes of an array
swapped = np.swapaxes(a, 0, 1)
print("Swapped axes array:\n", swapped)

# Roll the specified axis backwards, until it lies in a given position
rolled = np.rollaxis(a, 1, 0)
print("Rolled axis array:\n", rolled)

# Move axes to new positions in the array
moved = np.moveaxis(a, 0, 1)
print("Moved axis array:\n", moved)

# Add a new axis at the specified position in the array
expanded = np.expand_dims(a, axis=0)
print("Expanded dims array:\n", expanded)

# Remove axes of length one from the shape of an array
squeezed = np.squeeze(expanded)
print("Squeezed array:\n", squeezed)

# Convert input to a 1D array, if not already
atleast_1d = np.atleast_1d(a)
print("At least 1D array:\n", atleast_1d)

# Convert input to a 2D array, if not already
atleast_2d = np.atleast_2d([1, 2, 3])
print("At least 2D array:\n", atleast_2d)

# Convert input to a 3D array, if not already
atleast_3d = np.atleast_3d([1, 2, 3])
print("At least 3D array:\n", atleast_3d)

# Stack arrays along a new axis
stacked = np.stack((a, a))
print("Stacked array:\n", stacked)

# Join arrays along an existing axis
concatenated = np.concatenate((a, a), axis=0)
print("Concatenated array:\n", concatenated)

# Stack 1D arrays as columns into a 2D array
column_stacked = np.column_stack((a[0], a[1]))
print("Column stacked array:\n", column_stacked)

# Stack 1D arrays as rows into a 2D array
row_stacked = np.row_stack((a[0], a[1]))
print("Row stacked array:\n", row_stacked)

# Stack arrays horizontally (column-wise)
hstacked = np.hstack((a, a))
print("Horizontally stacked array:\n", hstacked)

# Stack arrays vertically (row-wise)
vstacked = np.vstack((a, a))
print("Vertically stacked array:\n", vstacked)

# Stack arrays along the third axis (depth)
dstacked = np.dstack((a, a))
print("Depth stacked array:\n", dstacked)

# Split an array into multiple sub-arrays
split = np.split(a, 2)
print("Split array:\n", split)

# Split an array into multiple sub-arrays, allowing unequal sizes
array_split = np.array_split(a, 3)
print("Array split:\n", array_split)

# Construct an array by repeating a smaller array multiple times
tiled = np.tile(a, (2, 1))
print("Tiled array:\n", tiled)

# Repeat elements of an array
repeated = np.repeat(a, 2, axis=0)
print("Repeated array:\n", repeated)

# Array creation functions
created_array = np.array([1, 2, 3])
print("Created array:\n", created_array)

# Create arrays with zeros, ones, empty, full, eye, identity
zeros_array = np.zeros((2, 2))
print("Zeros array:\n", zeros_array)

ones_array = np.ones((2, 2))
print("Ones array:\n", ones_array)

empty_array = np.empty((2, 2))
print("Empty array:\n", empty_array)

full_array = np.full((2, 2), 7)
print("Full array:\n", full_array)

eye_array = np.eye(2)
print("Eye array:\n", eye_array)

identity_array = np.identity(2)
print("Identity array:\n", identity_array)

# Create arrays with arange, linspace, random.rand, random.randn
arange_array = np.arange(5)
print("Arange array:\n", arange_array)

linspace_array = np.linspace(0, 10, 5)
print("Linspace array:\n", linspace_array)

random_rand_array = np.random.rand(2, 2)
print("Random rand array:\n", random_rand_array)

random_randn_array = np.random.randn(2, 2)
print("Random randn array:\n", random_randn_array)

# Create arrays with fromiter, frombuffer, asarray, copy
fromiter_array = np.fromiter(range(5), dtype=int)
print("Fromiter array:\n", fromiter_array)

frombuffer_array = np.frombuffer(b'hello world', dtype='S1')
print("Frombuffer array:\n", frombuffer_array)

asarray_array = np.asarray([1, 2, 3])
print("Asarray array:\n", asarray_array)

copy_array = np.copy(a)
print("Copy array:\n", copy_array)

# Basic array operations
added = np.add(a, a)
print("Added array:\n", added)

subtracted = np.subtract(a, a)
print("Subtracted array:\n", subtracted)

multiplied = np.multiply(a, a)
print("Multiplied array:\n", multiplied)

divided = np.divide(a, a)
print("Divided array:\n", divided)

floor_divided = np.floor_divide(a, 2)
print("Floor divided array:\n", floor_divided)

mod_array = np.mod(a, 2)
print("Mod array:\n", mod_array)

remainder_array = np.remainder(a, 2)
print("Remainder array:\n", remainder_array)

power_array = np.power(a, 2)
print("Power array:\n", power_array)

float_power_array = np.float_power(a, 2)
print("Float power array:\n", float_power_array)

fmod_array = np.fmod(a, 2)
print("Fmod array:\n", fmod_array)

negative_array = np.negative(a)
print("Negative array:\n", negative_array)

sign_array = np.sign(a)
print("Sign array:\n", sign_array)

reciprocal_array = np.reciprocal(a, where=a!=0)
print("Reciprocal array:\n", reciprocal_array)

# Array aggregation functions
summed = np.sum(a)
print("Summed array:\n", summed)

prod_array = np.prod(a)
print("Product array:\n", prod_array)

cumsum_array = np.cumsum(a)
print("Cumulative sum array:\n", cumsum_array)

cumprod_array = np.cumprod(a)
print("Cumulative product array:\n", cumprod_array)

min_array = np.min(a)
print("Min array:\n", min_array)

max_array = np.max(a)
print("Max array:\n", max_array)

argmin_array = np.argmin(a)
print("Argmin array:\n", argmin_array)

argmax_array = np.argmax(a)
print("Argmax array:\n", argmax_array)

mean_array = np.mean(a)
print("Mean array:\n", mean_array)

std_array = np.std(a)
print("Standard deviation array:\n", std_array)

var_array = np.var(a)
print("Variance array:\n", var_array)

median_array = np.median(a)
print("Median array:\n", median_array)

percentile_array = np.percentile(a, 50)
print("Percentile array:\n", percentile_array)

# Statistical operations
histogram = np.histogram(a)
print("Histogram:\n", histogram)

corrcoef_array = np.corrcoef(a)
print("Correlation coefficient array:\n", corrcoef_array)

cov_array = np.cov(a)
print("Covariance array:\n", cov_array)

diff_array = np.diff(a)
print("Difference array:\n", diff_array)

gradient_array = np.gradient(a)
print("Gradient array:\n", gradient_array)

quantile_array = np.quantile(a, 0.5)
print("Quantile array:\n", quantile_array)

# Manipulation and indexing functions
taken = np.take(a, [0, 2])
print("Taken elements:\n", taken)

put_array = np.put(a, [0, 2], [9, 9])
print("Array after put:\n", a)

# Corrected np.choose function call
choices = np.array([[10, 20, 30], [40, 50, 60]])
chosen = np.choose([0, 1, 0], choices)
print("Chosen elements:\n", chosen)

where_array = np.where(a > 2)
print("Where array:\n", where_array)

nonzero_array = np.nonzero(a)
print("Nonzero array:\n", nonzero_array)

extracted = np.extract(a > 2, a)
print("Extracted elements:\n", extracted)

putmask_array = np.putmask(a, a > 2, 0)
print("Array after putmask:\n", a)

deleted = np.delete(a, [0, 2])
print("Deleted elements array:\n", deleted)

inserted = np.insert(a, 1, [9, 9])
print("Inserted elements array:\n", inserted)

# Array properties
shape = a.shape
print("Shape of array:\n", shape)

ndim = a.ndim
print("Number of dimensions:\n", ndim)

size = a.size
print("Size of array:\n", size)

dtype = a.dtype
print("Data type of array:\n", dtype)

itemsize = a.itemsize
print("Item size of array:\n", itemsize)

transpose = a.T
print("Transposed array:\n", transpose)

flatten = a.flatten()
print("Flattened array:\n", flatten)

reshape = a.reshape((3, 2))
print("Reshaped array:\n", reshape)

ravel = a.ravel()
print("Raveled array:\n", ravel)

# Matrix operations
dot_product = np.dot(a, a.T)
print("Dot product:\n", dot_product)

vdot_product = np.vdot(a, a)
print("Vdot product:\n", vdot_product)

cross_product = np.cross(a[0], a[1])
print("Cross product:\n", cross_product)

inner_product = np.inner(a, a)
print("Inner product:\n", inner_product)

outer_product = np.outer(a, a)
print("Outer product:\n", outer_product)

matmul_product = np.matmul(a, a.T)
print("Matmul product:\n", matmul_product)

inv_matrix = np.linalg.inv(np.array([[1, 2], [3, 4]]))
print("Inverse matrix:\n", inv_matrix)

det_matrix = np.linalg.det(np.array([[1, 2], [3, 4]]))
print("Determinant of matrix:\n", det_matrix)

eig_values, eig_vectors = np.linalg.eig(np.array([[1, 2], [3, 4]]))
print("Eigenvalues:\n", eig_values)
print("Eigenvectors:\n", eig_vectors)

# Matrix functions
trace = np.trace(a)
print("Trace of array:\n", trace)

eye_matrix = np.eye(2)
print("Eye matrix:\n", eye_matrix)

identity_matrix = np.identity(2)
print("Identity matrix:\n", identity_matrix)

diag_matrix = np.diag([1, 2, 3])
print("Diagonal matrix:\n", diag_matrix)

kron_product = np.kron(a, a)
print("Kronecker product:\n", kron_product)

