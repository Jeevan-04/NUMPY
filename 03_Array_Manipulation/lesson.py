import numpy as np

# Reshape
a = np.array([[1, 2, 3], [4, 5, 6]])
print(np.reshape(a, (3, 2)))

# Ravel
print(a.ravel())

# Flatten
print(a.flatten())

# Transpose
print(a.T)

# np.transpose
print(np.transpose(a))

# Swapaxes
print(np.swapaxes(a, 0, 1))

# Moveaxis
b = np.zeros((2, 3, 4))
print(np.moveaxis(b, 0, -1))

# Expand_dims
c = np.array([1, 2, 3])
print(np.expand_dims(c, axis=0))

# Squeeze
d = np.array([[[1], [2], [3]]])
print(np.squeeze(d))

# Concatenate
e = np.array([1, 2])
f = np.array([3, 4])
print(np.concatenate((e, f)))

# Stack
print(np.stack((e, f), axis=0))

# Hstack
print(np.hstack((e, f)))

# Vstack
print(np.vstack((e, f)))

# Dstack
print(np.dstack((e, f)))

# Split
g = np.array([1, 2, 3, 4])
print(np.split(g, 2))

# Hsplit
h = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(np.hsplit(h, 2))

# Vsplit
print(np.vsplit(h, 2))

# Dsplit
i = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(np.dsplit(i, 2))

# Array_split
print(np.array_split(g, 3))

# Resize
print(np.resize(g, (2, 3)))

# Tile
j = np.array([1, 2, 3])
print(np.tile(j, 2))

# Repeat
print(np.repeat(j, 2))

# Array Operations
k = np.array([1, 2, 3])
l = np.array([4, 5, 6])
print(np.add(k, l))
print(np.subtract(k, l))
print(np.multiply(k, l))
print(np.divide(k, l))

# Stacking
print(np.vstack((e, f)))
print(np.hstack((e, f)))
print(np.dstack((e, f)))

# Splitting
print(np.split(g, 2))
print(np.array_split(g, 3))
print(np.hsplit(h, 2))
print(np.vsplit(h, 2))

# Concatenation
print(np.concatenate((e, f)))
print(np.append(e, f))

# Sorting
m = np.array([3, 1, 2])
print(np.sort(m))
print(np.argsort(m))

# Slicing
n = np.array([1, 2, 3, 4])
print(n[1:3])
print(np.take(n, [0, 2]))
n.put([0, 2], [5, 6])
print(n)

