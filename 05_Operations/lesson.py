import numpy as np

# Sum of array elements
arr = np.array([1, 2, 3, 4])
print("Sum:", np.sum(arr))

# Product of array elements
print("Product:", np.prod(arr))

# Cumulative sum of array elements
print("Cumulative Sum:", np.cumsum(arr))

# Cumulative product of array elements
print("Cumulative Product:", np.cumprod(arr))

# Minimum and maximum of array elements
print("Minimum:", np.min(arr))
print("Maximum:", np.max(arr))

# Indices of minimum and maximum elements
print("Index of Minimum:", np.argmin(arr))
print("Index of Maximum:", np.argmax(arr))

# Mean and median of array elements
print("Mean:", np.mean(arr))
print("Median:", np.median(arr))

# Standard deviation and variance of array elements
print("Standard Deviation:", np.std(arr))
print("Variance:", np.var(arr))

# Clip (limit) the values in an array
print("Clipped Array:", np.clip(arr, 2, 3))

# Power and float power of array elements
print("Power:", np.power(arr, 2))
print("Float Power:", np.float_power(arr, 2))

# Subtract, multiply, divide, and modulus of array elements
print("Subtract:", np.subtract(arr, 1))
print("Multiply:", np.multiply(arr, 2))
print("Divide:", np.divide(arr, 2))
print("Modulus:", np.mod(arr, 2))

# Add array elements
print("Add:", np.add(arr, 2))

# Logical operations on array elements
print("Logical AND:", np.logical_and(arr > 1, arr < 4))
print("Logical OR:", np.logical_or(arr < 2, arr > 3))
print("Logical NOT:", np.logical_not(arr))
print("Logical XOR:", np.logical_xor(arr > 1, arr < 4))

# Comparison operations on array elements
print("Equal:", np.equal(arr, 2))
print("Not Equal:", np.not_equal(arr, 2))
print("Greater:", np.greater(arr, 2))
print("Greater Equal:", np.greater_equal(arr, 2))
print("Less:", np.less(arr, 2))
print("Less Equal:", np.less_equal(arr, 2))

# Exponential and logarithmic functions
print("Exponential:", np.exp(arr))
print("Natural Logarithm:", np.log(arr))
print("Logarithm base 10:", np.log10(arr))
print("Logarithm base 2:", np.log2(arr))

# Trigonometric functions
print("Sine:", np.sin(arr))
print("Cosine:", np.cos(arr))
print("Tangent:", np.tan(arr))

# Inverse trigonometric functions
print("Arcsine:", np.arcsin(arr / 4))
print("Arccosine:", np.arccos(arr / 4))
print("Arctangent:", np.arctan(arr))

# Hyperbolic functions
print("Hyperbolic Sine:", np.sinh(arr))
print("Hyperbolic Cosine:", np.cosh(arr))
print("Hyperbolic Tangent:", np.tanh(arr))

# Additional array operations
# Sorting array elements
print("Sorted Array:", np.sort(arr))

# Unique elements in array
print("Unique Elements:", np.unique(arr))

# Concatenating arrays
arr2 = np.array([5, 6])
print("Concatenated Array:", np.concatenate((arr, arr2)))

# Reshaping array
reshaped_arr = np.reshape(arr, (2, 2))
print("Reshaped Array (2x2):", reshaped_arr)

# Matrix multiplication and other matrix operations
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
print("Matrix Multiplication:", np.matmul(matrix1, matrix2))
print("Matrix Multiplication (@ operator):", matrix1 @ matrix2)
print("Matrix Determinant:", np.linalg.det(matrix1))
print("Matrix Inverse:", np.linalg.inv(matrix1))
print("Matrix Transpose:", np.transpose(matrix1))

# Additional matrix operations
# Matrix rank
print("Matrix Rank:", np.linalg.matrix_rank(matrix1))

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix1)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)

# Matrix trace (sum of diagonal elements)
print("Matrix Trace:", np.trace(matrix1))

# Matrix dot product
print("Matrix Dot Product:", np.dot(matrix1, matrix2))

# Matrix element-wise multiplication
print("Element-wise Multiplication:", np.multiply(matrix1, matrix2))

