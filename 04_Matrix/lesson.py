import numpy as np

# Create a 2D matrix
A = np.matrix([[1, 2], [3, 4]])
print("2D Matrix A:\n", A)

# Create a 3D matrix
B = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("3D Matrix B:\n", B)

# Create a 4D matrix
C = np.random.rand(2, 2, 2, 2)
print("4D Matrix C:\n", C)

# .A - Return as an array
array_A = A.A
print("Array A:\n", array_A)

# .T - Transpose of the matrix
transpose_A = A.T
print("Transpose of A:\n", transpose_A)

# .I - Inverse of the matrix
inverse_A = A.I
print("Inverse of A:\n", inverse_A)

# .H - Conjugate transpose of the matrix
conjugate_transpose_A = A.H
print("Conjugate Transpose of A:\n", conjugate_transpose_A)

# np.diag() - Extract a diagonal or construct a diagonal array
diag_A = np.diag([1, 2, 3])
print("Diagonal matrix from [1, 2, 3]:\n", diag_A)
diagonal_A = np.diagonal(A)
print("Diagonal of A:\n", diagonal_A)

# np.tri() - Construct a matrix filled with ones at and below the given diagonal
tri_A = np.tri(3, 3)
print("Triangular matrix:\n", tri_A)

# np.tril() - Lower triangle of an array
tril_A = np.tril(A)
print("Lower triangle of A:\n", tril_A)

# np.triu() - Upper triangle of an array
triu_A = np.triu(A)
print("Upper triangle of A:\n", triu_A)

# np.dot() - Dot product of two arrays
dot_product = np.dot(A, A)
print("Dot product of A with A:\n", dot_product)

# .dot() - Dot product method
dot_product_method = A.dot(A)
print("Dot product method of A with A:\n", dot_product_method)

# .A - Return as an array
array_A_again = A.A
print("Array A again:\n", array_A_again)

# .tolist() - Return the matrix as a nested list
list_A = A.tolist()
print("List A:\n", list_A)

# @ operator - Matrix multiplication
matrix_multiplication = A @ A
print("Matrix multiplication of A with A:\n", matrix_multiplication)

# np.trace() - Sum along diagonals
trace_A = np.trace(A)
print("Trace of A:\n", trace_A)

# np.linalg.det() - Compute the determinant of an array
det_A = np.linalg.det(A)
print("Determinant of A:\n", det_A)

# np.linalg.matrix_rank() - Return matrix rank
rank_A = np.linalg.matrix_rank(A)
print("Rank of A:\n", rank_A)

# np.linalg.inv() - Compute the (multiplicative) inverse of a matrix
inv_A = np.linalg.inv(A)
print("Inverse of A using np.linalg.inv:\n", inv_A)

# np.linalg.eig() - Compute the eigenvalues and right eigenvectors of a square array
eigvals_A, eigvecs_A = np.linalg.eig(A)
print("Eigenvalues of A:\n", eigvals_A)
print("Eigenvectors of A:\n", eigvecs_A)

# np.linalg.svd() - Singular Value Decomposition
U, s, V = np.linalg.svd(A)
print("Singular Value Decomposition of A:\nU:\n", U, "\ns:\n", s, "\nV:\n", V)

# np.matmul() - Matrix product of two arrays
matmul_A = np.matmul(A, A)
print("Matrix product of A with A using np.matmul:\n", matmul_A)

# np.inner() - Inner product of two arrays
inner_product = np.inner(A, A)
print("Inner product of A with A:\n", inner_product)

# np.outer() - Compute the outer product of two vectors
outer_product = np.outer(np.array([1, 2]), np.array([3, 4]))
print("Outer product of [1, 2] and [3, 4]:\n", outer_product)

# np.vdot() - Return the dot product of two vectors
vdot_product = np.vdot(A, A)
print("Dot product of A with A using np.vdot:\n", vdot_product)