import numpy as np

# Example of np.broadcast()
a = np.array([1, 2, 3])
b = np.array([[4], [5], [6]])
broadcast = np.broadcast(a, b)
print("Broadcast shape:", broadcast.shape)
# Output: (3, 3)

# Example of np.broadcast_arrays()
a = np.array([1, 2, 3])
b = np.array([[4], [5], [6]])
broadcasted_arrays = np.broadcast_arrays(a, b)
print("Broadcasted arrays:")
for arr in broadcasted_arrays:
    print(arr)
# Output: 
# [[1 2 3]
#  [1 2 3]
#  [1 2 3]]
# [[4 4 4]
#  [5 5 5]
#  [6 6 6]]

# Example of np.broadcast_to()
a = np.array([1, 2, 3])
broadcasted = np.broadcast_to(a, (3, 3))
print("Broadcasted to shape (3, 3):")
print(broadcasted)
# Output:
# [[1 2 3]
#  [1 2 3]
#  [1 2 3]]

# Example of np.reshape() with broadcasting support
a = np.array([1, 2, 3, 4])
reshaped = np.reshape(a, (2, 2))
print("Reshaped array:")
print(reshaped)
# Output:
# [[1 2]
#  [3 4]]

# Example of np.newaxis
a = np.array([1, 2, 3])
newaxis_array = a[:, np.newaxis]
print("Array with new axis:")
print(newaxis_array)
# Output:
# [[1]
#  [2]
#  [3]]

# Example of np.expand_dims()
a = np.array([1, 2, 3])
expanded_array = np.expand_dims(a, axis=1)
print("Expanded array:")
print(expanded_array)
# Output:
# [[1]
#  [2]
#  [3]]

# Example of np.squeeze()
a = np.array([[[1], [2], [3]]])
squeezed_array = np.squeeze(a)
print("Squeezed array:")
print(squeezed_array)
# Output:
# [1 2 3]
