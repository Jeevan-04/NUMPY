import numpy as np

# Basic Linear Algebra
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Dot product
print("Dot product:\n", np.dot(a, b))

# Vector dot product
print("Vector dot product:\n", np.vdot(a, b))

# Inner product
print("Inner product:\n", np.inner(a, b))

# Outer product
print("Outer product:\n", np.outer(a, b))

# Matrix multiplication
print("Matrix multiplication:\n", np.matmul(a, b))

# Tensor dot product
print("Tensor dot product:\n", np.tensordot(a, b))

# Kronecker product
print("Kronecker product:\n", np.kron(a, b))

# Cross product
a1 = np.array([1, 2, 3])
b1 = np.array([4, 5, 6])
print("Cross product:\n", np.cross(a1, b1))

# Matrix Operations
# Matrix rank
print("Matrix rank:\n", np.linalg.matrix_rank(a))

# Matrix inverse
print("Matrix inverse:\n", np.linalg.inv(a))

# Pseudo-inverse
print("Pseudo-inverse:\n", np.linalg.pinv(a))

# Determinant
print("Determinant:\n", np.linalg.det(a))

# Norm
print("Norm:\n", np.linalg.norm(a))

# Matrix power
print("Matrix power:\n", np.linalg.matrix_power(a, 2))

# Trace
print("Trace:\n", np.trace(a))

# Multi dot
print("Multi dot:\n", np.linalg.multi_dot([a, b, a]))

# Decompositions
# Cholesky decomposition
print("Cholesky decomposition:\n", np.linalg.cholesky(np.array([[1, 2], [2, 5]])))

# QR decomposition
q, r = np.linalg.qr(a)
print("QR decomposition:\nQ:\n", q, "\nR:\n", r)

# Singular Value Decomposition
u, s, vh = np.linalg.svd(a)
print("Singular Value Decomposition:\nU:\n", u, "\nS:\n", s, "\nVH:\n", vh)

# Eigenvalues and Eigenvectors
eigvals, eigvecs = np.linalg.eig(a)
print("Eigenvalues:\n", eigvals)
print("Eigenvectors:\n", eigvecs)

# Hermitian or symmetric matrices
print("Eigenvalues of Hermitian matrix:\n", np.linalg.eigh(a))

# Eigenvalues only
print("Eigenvalues only:\n", np.linalg.eigvals(a))

# Eigenvalues of Hermitian matrix only
print("Eigenvalues of Hermitian matrix only:\n", np.linalg.eigvalsh(a))

# LU decomposition (Note: numpy does not have a direct LU decomposition function)
# Using scipy for LU decomposition
from scipy.linalg import lu
p, l, u = lu(a)
print("LU decomposition:\nP:\n", p, "\nL:\n", l, "\nU:\n", u)

# Least squares solution
# Fixing the dimension mismatch by using a compatible vector
b2 = np.array([1, 2])
print("Least squares solution:\n", np.linalg.lstsq(a, b2, rcond=None))

# Solving Linear Systems
# Solve linear system
print("Solve linear system:\n", np.linalg.solve(a, np.array([1, 2])))

# Eigenvalues and Eigenvectors
print("Eigenvalues and Eigenvectors:\n", np.linalg.eig(a))

# Singular Value Decomposition
print("Singular Value Decomposition:\n", np.linalg.svd(a))

# Condition Numbers
print("Condition Numbers:\n", np.linalg.cond(a))

# Matrix Factorization
print("Cholesky Factorization:\n", np.linalg.cholesky(np.array([[1, 2], [2, 5]])))
print("QR Factorization:\n", np.linalg.qr(a))

# Utility Functions
print("Norm:\n", np.linalg.norm(a))
print("Determinant:\n", np.linalg.det(a))
print("Matrix rank:\n", np.linalg.matrix_rank(a))
print("Multi dot:\n", np.linalg.multi_dot([a, b, a]))
