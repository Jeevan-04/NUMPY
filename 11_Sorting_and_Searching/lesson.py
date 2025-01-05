import numpy as np

# Sorting Functions
arr = np.array([3, 1, 2, 5, 4])

# np.sort()
sorted_arr = np.sort(arr)
print("np.sort:", sorted_arr)

# np.argsort()
argsorted_indices = np.argsort(arr)
print("np.argsort:", argsorted_indices)

# np.lexsort()
names = ('John', 'Jane', 'Doe', 'Alice')
ages = (25, 30, 20, 22)
lexsorted_indices = np.lexsort((ages, names))
print("np.lexsort:", lexsorted_indices)

# np.partition()
partitioned_arr = np.partition(arr, 3)
print("np.partition:", partitioned_arr)

# np.argpartition()
argpartitioned_indices = np.argpartition(arr, 3)
print("np.argpartition:", argpartitioned_indices)

# Searching Functions
arr = np.array([1, 2, 3, 4, 5])

# np.searchsorted()
searchsorted_index = np.searchsorted(arr, 3)
print("np.searchsorted:", searchsorted_index)

# np.where()
where_indices = np.where(arr > 3)
print("np.where:", where_indices)

# np.nonzero()
nonzero_indices = np.nonzero(arr)
print("np.nonzero:", nonzero_indices)

# np.argmax()
argmax_index = np.argmax(arr)
print("np.argmax:", argmax_index)

# np.argmin()
argmin_index = np.argmin(arr)
print("np.argmin:", argmin_index)

# np.nanargmax()
arr_with_nan = np.array([1, np.nan, 3, 2])
nanargmax_index = np.nanargmax(arr_with_nan)
print("np.nanargmax:", nanargmax_index)

# np.nanargmin()
nanargmin_index = np.nanargmin(arr_with_nan)
print("np.nanargmin:", nanargmin_index)

# np.extract()
condition = arr > 2
extracted_elements = np.extract(condition, arr)
print("np.extract:", extracted_elements)

# np.flatnonzero()
flatnonzero_indices = np.flatnonzero(arr)
print("np.flatnonzero:", flatnonzero_indices)

# Set Operations (for Searching Unique Elements)
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([3, 4, 5, 6])

# np.unique()
unique_elements = np.unique(arr1)
print("np.unique:", unique_elements)

# np.in1d()
in1d_result = np.in1d(arr1, arr2)
print("np.in1d:", in1d_result)

# np.intersect1d()
intersect1d_result = np.intersect1d(arr1, arr2)
print("np.intersect1d:", intersect1d_result)

# np.setdiff1d()
setdiff1d_result = np.setdiff1d(arr1, arr2)
print("np.setdiff1d:", setdiff1d_result)

# np.setxor1d()
setxor1d_result = np.setxor1d(arr1, arr2)
print("np.setxor1d:", setxor1d_result)

# np.union1d()
union1d_result = np.union1d(arr1, arr2)
print("np.union1d:", union1d_result)
