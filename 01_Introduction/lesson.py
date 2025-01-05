#########################################
# IMPORTS
#########################################
import numpy as np

# Print the version of NumPy
print(np.__version__)
print()

# Create a 1D array
a = np.array([1, 2, 3, 4, 5])
print(a)
print("No. of dimensions: ", a.ndim)  # ndim returns the number of dimensions
print("Shape of array: ", a.shape)    # shape returns the dimensions of the array
print("Size of array: ", a.size)      # size returns the total number of elements
print("Array stores elements of type: ", a.dtype)  # dtype returns the data type of the elements
print("Item size of array: ", a.itemsize)  # itemsize returns the size in bytes of each element
print()

# Create a 2D array
b = np.array([[1, 2, 3], [4, 5, 6]])
print(b)
print("No. of dimensions: ", b.ndim)  # ndim returns the number of dimensions
print("Shape of array: ", b.shape)    # shape returns the dimensions of the array
print("Size of array: ", b.size)      # size returns the total number of elements
print("Array stores elements of type: ", b.dtype)  # dtype returns the data type of the elements
print("Item size of array: ", b.itemsize)  # itemsize returns the size in bytes of each element
print()

# Create a 3D array
c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(c)
print("No. of dimensions: ", c.ndim)  # ndim returns the number of dimensions
print("Shape of array: ", c.shape)    # shape returns the dimensions of the array
print("Size of array: ", c.size)      # size returns the total number of elements
print("Array stores elements of type: ", c.dtype)  # dtype returns the data type of the elements
print("Item size of array: ", c.itemsize)  # itemsize returns the size in bytes of each element
print()

# Create a 1D array from a tuple
d = np.array((1, 2))
print(d)
print("No. of dimensions: ", d.ndim)  # ndim returns the number of dimensions
print("Shape of array: ", d.shape)    # shape returns the dimensions of the array
print("Size of array: ", d.size)      # size returns the total number of elements
print("Array stores elements of type: ", d.dtype)  # dtype returns the data type of the elements
print("Item size of array: ", d.itemsize)  # itemsize returns the size in bytes of each element
print()

# Create an array of sets
e = np.array([{1, 2}, {2, 4}, {3, 6}])
print(e)
print("No. of dimensions: ", e.ndim)  # ndim returns the number of dimensions
print("Array stores elements of type: ", e.dtype)  # dtype returns the data type of the elements
print()

# Create an array of strings
f = np.array(['namaste', 'Hello'])
print(f)
print("Size of array: ", f.size)      # size returns the total number of elements
print("Array stores elements of type: ", f.dtype)  # dtype returns the data type of the elements
print()

# Create an array of boolean values
g = np.array([True, False])
print(g)
print("Size of array: ", g.size)      # size returns the total number of elements
print("Array stores elements of type: ", g.dtype)  # dtype returns the data type of the elements
print()

# Create an array of complex numbers
h = np.array([1+2j, 3+4j, 5+6*1j])
print(h)
print("Size of array: ", h.size)      # size returns the total number of elements
print("Array stores elements of type: ", h.dtype)  # dtype returns the data type of the elements
print()

# Create an array of floats
i = np.array([1., 2., 3.])
print(i)
print("Size of array: ", i.size)      # size returns the total number of elements
print("Array stores elements of type: ", i.dtype)  # dtype returns the data type of the elements
print()

# Create arrays using various functions
zeros_array = np.zeros((2, 3))
print("Zeros array:\n", zeros_array)
print()

ones_array = np.ones((2, 3))
print("Ones array:\n", ones_array)
print()

empty_array = np.empty((2, 3))
print("Empty array:\n", empty_array)
print()

full_array = np.full((2, 3), 7)
print("Full array:\n", full_array)
print()

arange_array = np.arange(10)
print("Arange array:\n", arange_array)
print()

linspace_array = np.linspace(0, 1, 5)
print("Linspace array:\n", linspace_array)
print()

rand_array = np.random.rand(2, 3)
print("Random array:\n", rand_array)
print()

randn_array = np.random.randn(2, 3)
print("Random normal array:\n", randn_array)
print()

eye_array = np.eye(3)
print("Eye array:\n", eye_array)
print()

identity_array = np.identity(3)
print("Identity array:\n", identity_array)
print()

# Type conversion
float_array = np.array([1, 2, 3], dtype=np.float32)
int_array = float_array.astype(np.int64)
print("Converted to int array:\n", int_array)
print()

cast_array = np.asarray(int_array, dtype=np.float32)
print("Cast to float array:\n", cast_array)
print()

# Convert to boolean
bool_array = np.array([0, 1, 2, 3], dtype=bool)
print("Converted to boolean array:\n", bool_array)
print()

# Convert to string
str_array = np.array([1, 2, 3], dtype=str)
print("Converted to string array:\n", str_array)
print()

# Convert to complex
complex_array = np.array([1, 2, 3], dtype=complex)
print("Converted to complex array:\n", complex_array)
print()

# Check subdtype
print("Is float32 a subdtype of float? ", np.issubdtype(np.float32, float))
print()

# Promote types
promoted_type = np.promote_types(np.int32, np.float32)
print("Promoted type of int32 and float32: ", promoted_type)
print()

# Can cast
print("Can cast float32 to int32 safely? ", np.can_cast(np.float32, np.int32))
print()

# Additional array attributes
print("Item size of array 'a': ", a.itemsize)
print("Transpose of array 'b':\n", b.T)