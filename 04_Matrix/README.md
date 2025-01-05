# Numpy Matrix Operations

This document provides examples of various matrix operations using the Numpy library in Python.

## Table of Contents

1. [Introduction](#introduction)
2. [Matrix Creation](#matrix-creation)
3. [Matrix Operations](#matrix-operations)
    - [.A - Return as an array](#a---return-as-an-array)
    - [.T - Transpose of the matrix](#t---transpose-of-the-matrix)
    - [.I - Inverse of the matrix](#i---inverse-of-the-matrix)
    - [.H - Conjugate transpose of the matrix](#h---conjugate-transpose-of-the-matrix)
    - [np.diag() - Diagonal array](#npdiag---diagonal-array)
    - [np.tri() - Triangular matrix](#nptri---triangular-matrix)
    - [np.tril() - Lower triangle](#nptril---lower-triangle)
    - [np.triu() - Upper triangle](#nptriu---upper-triangle)
    - [np.dot() - Dot product](#npdot---dot-product)
    - [.dot() - Dot product method](#dot---dot-product-method)
    - [.tolist() - Nested list](#tolist---nested-list)
    - [@ operator - Matrix multiplication](#operator---matrix-multiplication)
    - [np.trace() - Sum along diagonals](#nptrace---sum-along-diagonals)
    - [np.linalg.det() - Determinant](#nplinalgdet---determinant)
    - [np.linalg.matrix_rank() - Matrix rank](#nplinalgmatrix_rank---matrix-rank)
    - [np.linalg.inv() - Inverse](#nplinalginv---inverse)
    - [np.linalg.eig() - Eigenvalues and eigenvectors](#nplinalgeig---eigenvalues-and-eigenvectors)
    - [np.linalg.svd() - Singular Value Decomposition](#nplinalgsvd---singular-value-decomposition)
    - [np.matmul() - Matrix product](#npmatmul---matrix-product)
    - [np.inner() - Inner product](#npinner---inner-product)
    - [np.outer() - Outer product](#npouter---outer-product)
    - [np.vdot() - Dot product](#npvdot---dot-product)
4. [Examples](#examples)

## Introduction

This document demonstrates how to perform various matrix operations using the Numpy library in Python. It includes examples of creating matrices, performing matrix multiplications, and using various matrix-related functions.

## Matrix Creation

- **2D Matrix**: `A = np.matrix([[1, 2], [3, 4]])`
- **3D Matrix**: `B = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])`
- **4D Matrix**: `C = np.random.rand(2, 2, 2, 2)`

## Matrix Operations

### .A - Return as an array
Converts a matrix to an array.
```python
array_A = A.A
```

### .T - Transpose of the matrix
Returns the transpose of the matrix.
```python
transpose_A = A.T
```

### .I - Inverse of the matrix
Returns the inverse of the matrix.
```python
inverse_A = A.I
```

### .H - Conjugate transpose of the matrix
Returns the conjugate transpose of the matrix.
```python
conjugate_transpose_A = A.H
```

### np.diag() - Diagonal array
Extracts a diagonal or constructs a diagonal array.
```python
diag_A = np.diag([1, 2, 3])
diagonal_A = np.diagonal(A)
```

### np.tri() - Triangular matrix
Constructs a matrix filled with ones at and below the given diagonal.
```python
tri_A = np.tri(3, 3)
```

### np.tril() - Lower triangle
Returns the lower triangle of an array.
```python
tril_A = np.tril(A)
```

### np.triu() - Upper triangle
Returns the upper triangle of an array.
```python
triu_A = np.triu(A)
```

### np.dot() - Dot product
Computes the dot product of two arrays.
```python
dot_product = np.dot(A, A)
```

### .dot() - Dot product method
Computes the dot product using the method.
```python
dot_product_method = A.dot(A)
```

### .tolist() - Nested list
Returns the matrix as a nested list.
```python
list_A = A.tolist()
```

### @ operator - Matrix multiplication
Performs matrix multiplication using the `@` operator.
```python
matrix_multiplication = A @ A
```

### np.trace() - Sum along diagonals
Computes the sum along the diagonals.
```python
trace_A = np.trace(A)
```

### np.linalg.det() - Determinant
Computes the determinant of an array.
```python
det_A = np.linalg.det(A)
```

### np.linalg.matrix_rank() - Matrix rank
Returns the rank of the matrix.
```python
rank_A = np.linalg.matrix_rank(A)
```

### np.linalg.inv() - Inverse
Computes the (multiplicative) inverse of a matrix.
```python
inv_A = np.linalg.inv(A)
```

### np.linalg.eig() - Eigenvalues and eigenvectors
Computes the eigenvalues and right eigenvectors of a square array.
```python
eigvals_A, eigvecs_A = np.linalg.eig(A)
```

### np.linalg.svd() - Singular Value Decomposition
Performs Singular Value Decomposition.
```python
U, s, V = np.linalg.svd(A)
```

### np.matmul() - Matrix product
Computes the matrix product of two arrays.
```python
matmul_A = np.matmul(A, A)
```

### np.inner() - Inner product
Computes the inner product of two arrays.
```python
inner_product = np.inner(A, A)
```

### np.outer() - Outer product
Computes the outer product of two vectors.
```python
outer_product = np.outer(np.array([1, 2]), np.array([3, 4]))
```

### np.vdot() - Dot product
Returns the dot product of two vectors.
```python
vdot_product = np.vdot(A, A)
```

## Examples

The examples provided in the `lesson.py` file demonstrate how to use these functions and operations with 2D, 3D, and 4D matrices.
