# Introduction to NumPy

## Table of Contents
1. [Introduction](#introduction)
2. [NumPy Version](#numpy-version)
3. [Creating Arrays](#creating-arrays)
4. [Array Attributes](#array-attributes)
5. [Array Data Types](#array-data-types)
6. [Type Conversion](#type-conversion)

## Introduction
NumPy is a powerful library for numerical computing in Python. It provides support for arrays, matrices, and many mathematical functions.

## NumPy Version
To check the version of NumPy installed, use:
```python
import numpy as np
print(np.__version__)
```

## Creating Arrays
NumPy arrays can be created using various functions:
```python
a = np.array([1, 2, 3, 4, 5])
b = np.zeros((2, 3))
c = np.ones((2, 3))
d = np.empty((2, 3))
e = np.full((2, 3), 7)
f = np.arange(10)
g = np.linspace(0, 1, 5)
h = np.random.rand(2, 3)
i = np.random.randn(2, 3)
j = np.eye(3)
k = np.identity(3)
```

## Array Attributes
- `ndim`: Returns the number of dimensions of the array.
  ```python
  a = np.array([[1, 2, 3], [4, 5, 6]])
  print(a.ndim)  # Output: 2
  ```
- `shape`: Returns the shape of the array (dimensions).
  ```python
  a = np.array([[1, 2, 3], [4, 5, 6]])
  print(a.shape)  # Output: (2, 3)
  ```
- `size`: Returns the total number of elements in the array.
  ```python
  a = np.array([[1, 2, 3], [4, 5, 6]])
  print(a.size)  # Output: 6
  ```
- `dtype`: Returns the data type of the elements in the array.
  ```python
  a = np.array([1, 2, 3])
  print(a.dtype)  # Output: int64
  ```
- `itemsize`: Returns the size in bytes of each element.
  ```python
  a = np.array([1, 2, 3])
  print(a.itemsize)  # Output: 8
  ```
- `T`: Returns the transpose of the array.
  ```python
  a = np.array([[1, 2, 3], [4, 5, 6]])
  print(a.T)
  # Output:
  # [[1 4]
  #  [2 5]
  #  [3 6]]
  ```

## Array Data Types
NumPy arrays can store elements of various data types, including integers, floats, complex numbers, strings, and boolean values.
```python
a = np.array([1, 2, 3], dtype=np.int64)
b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
c = np.array([1+2j, 3+4j], dtype=np.complex128)
d = np.array([True, False], dtype=np.bool_)
e = np.array(['a', 'b', 'c'], dtype=np.object)
f = np.array(['2021-01-01'], dtype=np.datetime64)
g = np.array([1, 2, 3], dtype=np.timedelta64)
```

## Type Conversion
NumPy provides functions for type conversion:
```python
a = np.array([1, 2, 3], dtype=np.float32)
b = a.astype(np.int64)
c = np.asarray(a, dtype=np.float32)
```
- `np.issubdtype()`: Checks if a data type is a sub-type of another.
- `np.promote_types()`: Returns the promoted data type.
- `np.can_cast()`: Checks if casting can be done safely.
