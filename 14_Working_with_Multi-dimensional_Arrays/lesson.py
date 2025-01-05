import numpy as np

# Create a sample array
arr = np.array([[1, 2, 3], [4, 5, 6]])

# np.reshape()
reshaped = np.reshape(arr, (3, 2))
print("Reshaped:\n", reshaped)

# np.flatten()
flattened = arr.flatten()
print("Flattened:", flattened)

# np.ravel()
raveled = np.ravel(arr)
print("Raveled:", raveled)

# np.transpose()
transposed = np.transpose(arr)
print("Transposed:\n", transposed)

# np.swapaxes()
swapped = np.swapaxes(arr, 0, 1)
print("Swapped axes:\n", swapped)

# np.moveaxis()
moved = np.moveaxis(arr, 0, 1)
print("Moved axes:\n", moved)

# np.rollaxis()
rolled = np.rollaxis(arr, 1, 0)
print("Rolled axes:\n", rolled)

# np.expand_dims()
expanded = np.expand_dims(arr, axis=0)
print("Expanded dims:\n", expanded)

# np.squeeze()
squeezed = np.squeeze(expanded)
print("Squeezed:\n", squeezed)

# np.hstack()
hstacked = np.hstack((arr, arr))
print("Hstacked:\n", hstacked)

# np.vstack()
vstacked = np.vstack((arr, arr))
print("Vstacked:\n", vstacked)

# np.dstack()
dstacked = np.dstack((arr, arr))
print("Dstacked:\n", dstacked)

# np.column_stack()
column_stacked = np.column_stack((arr, arr))
print("Column stacked:\n", column_stacked)

# np.row_stack()
row_stacked = np.row_stack((arr, arr))
print("Row stacked:\n", row_stacked)

# np.stack()
stacked = np.stack((arr, arr), axis=0)
print("Stacked:\n", stacked)

# np.split()
split = np.split(arr, 2)
print("Split:", split)

# np.array_split()
array_split = np.array_split(arr, 3)
print("Array split:", array_split)

# np.concatenate()
concatenated = np.concatenate((arr, arr), axis=0)
print("Concatenated:\n", concatenated)

# np.insert()
inserted = np.insert(arr, 1, 99, axis=0)
print("Inserted:\n", inserted)

# np.delete()
deleted = np.delete(arr, 1, axis=0)
print("Deleted:\n", deleted)

# np.append()
appended = np.append(arr, [[7, 8, 9]], axis=0)
print("Appended:\n", appended)

# np.resize()
resized = np.resize(arr, (3, 3))
print("Resized:\n", resized)

# np.unique()
unique = np.unique(arr)
print("Unique:", unique)

# np.meshgrid()
x = np.array([1, 2, 3])
y = np.array([4, 5])
xx, yy = np.meshgrid(x, y)
print("Meshgrid xx:\n", xx)
print("Meshgrid yy:\n", yy)

# np.ix_()
ix = np.ix_([0, 1], [1, 2])
print("Ix_:", arr[ix])

# np.ndindex()
print("Ndindex:")
for index in np.ndindex(arr.shape):
    print(index, arr[index])

# np.ndenumerate()
print("Ndenumerate:")
for index, value in np.ndenumerate(arr):
    print(index, value)

# np.broadcast()
broadcasted = np.broadcast(arr, arr)
print("Broadcast shape:", broadcasted.shape)

# np.broadcast_arrays()
broadcast_arrays = np.broadcast_arrays(arr, arr)
print("Broadcast arrays:", broadcast_arrays)

# np.broadcast_to()
broadcast_to = np.broadcast_to(arr, (2, 2, 3))
print("Broadcast to:\n", broadcast_to)

# np.r_[]
r_ = np.r_[arr, arr]
print("R_:\n", r_)

# np.c_[]
c_ = np.c_[arr, arr]
print("C_:\n", c_)

# np.mgrid[]
mgrid = np.mgrid[0:3, 0:3]
print("Mgrid:\n", mgrid)

# np.ogrid[]
ogrid = np.ogrid[0:3, 0:3]
print("Ogrid:\n", ogrid)
