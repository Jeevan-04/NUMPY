# Numpy Operations

This document provides examples of various numpy operations for arrays and matrices. The examples are implemented in the `lesson.py` file.

## Table of Contents

1. [Array Operations](#array-operations)
    - [Sum](#sum)
    - [Product](#product)
    - [Cumulative Sum](#cumulative-sum)
    - [Cumulative Product](#cumulative-product)
    - [Minimum and Maximum](#minimum-and-maximum)
    - [Indices of Minimum and Maximum](#indices-of-minimum-and-maximum)
    - [Mean and Median](#mean-and-median)
    - [Standard Deviation and Variance](#standard-deviation-and-variance)
    - [Clip](#clip)
    - [Power and Float Power](#power-and-float-power)
    - [Arithmetic Operations](#arithmetic-operations)
    - [Logical Operations](#logical-operations)
    - [Comparison Operations](#comparison-operations)
    - [Exponential and Logarithmic Functions](#exponential-and-logarithmic-functions)
    - [Trigonometric Functions](#trigonometric-functions)
    - [Inverse Trigonometric Functions](#inverse-trigonometric-functions)
    - [Hyperbolic Functions](#hyperbolic-functions)
    - [Sorting](#sorting)
    - [Unique Elements](#unique-elements)
    - [Concatenation](#concatenation)
    - [Reshaping](#reshaping)
2. [Matrix Operations](#matrix-operations)
    - [Matrix Multiplication](#matrix-multiplication)
    - [Matrix Determinant](#matrix-determinant)
    - [Matrix Inverse](#matrix-inverse)
    - [Matrix Transpose](#matrix-transpose)
    - [Matrix Rank](#matrix-rank)
    - [Eigenvalues and Eigenvectors](#eigenvalues-and-eigenvectors)
    - [Matrix Trace](#matrix-trace)
    - [Matrix Dot Product](#matrix-dot-product)
    - [Element-wise Multiplication](#element-wise-multiplication)

## Array Operations

### Sum
Calculate the sum of array elements.
```python
np.sum(arr)
```

### Product
Calculate the product of array elements.
```python
np.prod(arr)
```

### Cumulative Sum
Calculate the cumulative sum of array elements.
```python
np.cumsum(arr)
```

### Cumulative Product
Calculate the cumulative product of array elements.
```python
np.cumprod(arr)
```

### Minimum and Maximum
Find the minimum and maximum of array elements.
```python
np.min(arr)
np.max(arr)
```

### Indices of Minimum and Maximum
Find the indices of the minimum and maximum elements.
```python
np.argmin(arr)
np.argmax(arr)
```

### Mean and Median
Calculate the mean and median of array elements.
```python
np.mean(arr)
np.median(arr)
```

### Standard Deviation and Variance
Calculate the standard deviation and variance of array elements.
```python
np.std(arr)
np.var(arr)
```

### Clip
Clip (limit) the values in an array.
```python
np.clip(arr, 2, 3)
```

### Power and Float Power
Calculate the power and float power of array elements.
```python
np.power(arr, 2)
np.float_power(arr, 2)
```

### Arithmetic Operations
Perform arithmetic operations on array elements.
```python
np.subtract(arr, 1)
np.multiply(arr, 2)
np.divide(arr, 2)
np.mod(arr, 2)
np.add(arr, 2)
```

### Logical Operations
Perform logical operations on array elements.
```python
np.logical_and(arr > 1, arr < 4)
np.logical_or(arr < 2, arr > 3)
np.logical_not(arr)
np.logical_xor(arr > 1, arr < 4)
```

### Comparison Operations
Perform comparison operations on array elements.
```python
np.equal(arr, 2)
np.not_equal(arr, 2)
np.greater(arr, 2)
np.greater_equal(arr, 2)
np.less(arr, 2)
np.less_equal(arr, 2)
```

### Exponential and Logarithmic Functions
Calculate exponential and logarithmic functions of array elements.
```python
np.exp(arr)
np.log(arr)
np.log10(arr)
np.log2(arr)
```

### Trigonometric Functions
Calculate trigonometric functions of array elements.
```python
np.sin(arr)
np.cos(arr)
np.tan(arr)
```

### Inverse Trigonometric Functions
Calculate inverse trigonometric functions of array elements.
```python
np.arcsin(arr / 4)
np.arccos(arr / 4)
np.arctan(arr)
```

### Hyperbolic Functions
Calculate hyperbolic functions of array elements.
```python
np.sinh(arr)
np.cosh(arr)
np.tanh(arr)
```

### Sorting
Sort array elements.
```python
np.sort(arr)
```

### Unique Elements
Find unique elements in an array.
```python
np.unique(arr)
```

### Concatenation
Concatenate arrays.
```python
np.concatenate((arr, arr2))
```

### Reshaping
Reshape an array.
```python
np.reshape(arr, (2, 2))
```

## Matrix Operations

### Matrix Multiplication
Perform matrix multiplication.
```python
np.matmul(matrix1, matrix2)
matrix1 @ matrix2
```

### Matrix Determinant
Calculate the determinant of a matrix.
```python
np.linalg.det(matrix1)
```

### Matrix Inverse
Calculate the inverse of a matrix.
```python
np.linalg.inv(matrix1)
```

### Matrix Transpose
Transpose a matrix.
```python
np.transpose(matrix1)
```

### Matrix Rank
Calculate the rank of a matrix.
```python
np.linalg.matrix_rank(matrix1)
```

### Eigenvalues and Eigenvectors
Calculate the eigenvalues and eigenvectors of a matrix.
```python
eigenvalues, eigenvectors = np.linalg.eig(matrix1)
```

### Matrix Trace
Calculate the trace (sum of diagonal elements) of a matrix.
```python
np.trace(matrix1)
```

### Matrix Dot Product
Calculate the dot product of two matrices.
```python
np.dot(matrix1, matrix2)
```

### Element-wise Multiplication
Perform element-wise multiplication of two matrices.
```python
np.multiply(matrix1, matrix2)
