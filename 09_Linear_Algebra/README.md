# Linear Algebra with NumPy

This project demonstrates various linear algebra operations using NumPy.

## Table of Contents
1. [Basic Linear Algebra](#basic-linear-algebra)
2. [Matrix Operations](#matrix-operations)
3. [Decompositions](#decompositions)
4. [Solving Linear Systems](#solving-linear-systems)
5. [Eigenvalues and Eigenvectors](#eigenvalues-and-eigenvectors)
6. [Singular Value Decomposition](#singular-value-decomposition)
7. [Condition Numbers](#condition-numbers)
8. [Matrix Factorization](#matrix-factorization)
9. [Utility Functions](#utility-functions)

## Basic Linear Algebra
- `np.dot()`: Dot product of two arrays.
- `np.vdot()`: Dot product of two vectors.
- `np.inner()`: Inner product of two arrays.
- `np.outer()`: Outer product of two vectors.
- `np.matmul()`: Matrix product of two arrays.
- `np.tensordot()`: Tensor dot product along specified axes.
- `np.kron()`: Kronecker product of two arrays.
- `np.cross()`: Cross product of two vectors.

## Matrix Operations
- `np.linalg.matrix_rank()`: Rank of a matrix.
- `np.linalg.inv()`: Inverse of a matrix.
- `np.linalg.pinv()`: Pseudo-inverse of a matrix.
- `np.linalg.det()`: Determinant of a matrix.
- `np.linalg.norm()`: Norm of a matrix or vector.
- `np.linalg.matrix_power()`: Matrix raised to a power.
- `np.trace()`: Sum of the diagonal elements of a matrix.
- `np.linalg.multi_dot()`: Efficiently multiply two or more matrices.

## Decompositions
- `np.linalg.cholesky()`: Cholesky decomposition.
- `np.linalg.qr()`: QR decomposition.
- `np.linalg.svd()`: Singular Value Decomposition.
- `np.linalg.eig()`: Eigenvalues and right eigenvectors.
- `np.linalg.eigh()`: Eigenvalues and eigenvectors of a Hermitian or symmetric matrix.
- `np.linalg.eigvals()`: Eigenvalues of a matrix.
- `np.linalg.eigvalsh()`: Eigenvalues of a Hermitian or symmetric matrix.
- `scipy.linalg.lu()`: LU decomposition (using SciPy).

## Solving Linear Systems
- `np.linalg.solve()`: Solve a linear matrix equation.
- `np.linalg.lstsq()`: Least-squares solution to a linear matrix equation.

## Eigenvalues and Eigenvectors
- `np.linalg.eig()`: Compute the eigenvalues and right eigenvectors of a square array.
- `np.linalg.eigh()`: Eigenvalues and eigenvectors of a Hermitian or symmetric matrix.
- `np.linalg.eigvals()`: Compute the eigenvalues of a general matrix.
- `np.linalg.eigvalsh()`: Compute the eigenvalues of a Hermitian or symmetric matrix.

## Singular Value Decomposition
- `np.linalg.svd()`: Singular Value Decomposition.

## Condition Numbers
- `np.linalg.cond()`: Compute the condition number of a matrix.

## Matrix Factorization
- `np.linalg.cholesky()`: Cholesky decomposition.
- `np.linalg.qr()`: QR decomposition.

## Utility Functions
- `np.linalg.norm()`: Matrix or vector norm.
- `np.linalg.det()`: Determinant of a matrix.
- `np.linalg.matrix_rank()`: Rank of a matrix.
- `np.linalg.multi_dot()`: Efficiently multiply two or more matrices.
