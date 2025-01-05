# NumPy Reshaping and Array Operations

## Table of Contents
1. [Introduction](#introduction)
2. [Reshaping Functions](#reshaping-functions)
3. [Array Creation Functions](#array-creation-functions)
4. [Basic Array Operations](#basic-array-operations)
5. [Array Aggregation Functions](#array-aggregation-functions)
6. [Statistical Operations](#statistical-operations)
7. [Manipulation and Indexing Functions](#manipulation-and-indexing-functions)
8. [Array Properties](#array-properties)
9. [Matrix Operations](#matrix-operations)
10. [Matrix Functions](#matrix-functions)

## Introduction
This document provides examples and descriptions of various NumPy functions used for reshaping arrays, creating arrays, performing basic operations, aggregating data, statistical analysis, manipulation and indexing, and matrix operations.

## Reshaping Functions
- **np.reshape(a, new_shape)**: Reshape an array to a new shape.
- **np.resize(a, new_shape)**: Reshape an array, possibly repeating elements to fit the new size.
- **np.ravel(a)**: Flatten a multi-dimensional array into a 1D array.
- **np.flatten()**: Return a flattened 1D array.
- **np.transpose(a, axes=None)**: Transpose the dimensions of an array.
- **np.swapaxes(a, axis1, axis2)**: Swap two axes of an array.
- **np.rollaxis(a, axis, start=0)**: Roll the specified axis backwards, until it lies in a given position.
- **np.moveaxis(a, source, destination)**: Move axes to new positions in the array.
- **np.expand_dims(a, axis)**: Add a new axis at the specified position in the array.
- **np.squeeze(a, axis=None)**: Remove axes of length one from the shape of an array.
- **np.atleast_1d(a)**: Convert input to a 1D array, if not already.
- **np.atleast_2d(a)**: Convert input to a 2D array, if not already.
- **np.atleast_3d(a)**: Convert input to a 3D array, if not already.
- **np.stack(arrays, axis=0)**: Stack arrays along a new axis.
- **np.concatenate((a1, a2, ...), axis=0)**: Join arrays along an existing axis.
- **np.column_stack(tup)**: Stack 1D arrays as columns into a 2D array.
- **np.row_stack(tup)**: Stack 1D arrays as rows into a 2D array.
- **np.hstack((a1, a2, ...))**: Stack arrays horizontally (column-wise).
- **np.vstack((a1, a2, ...))**: Stack arrays vertically (row-wise).
- **np.dstack((a1, a2, ...))**: Stack arrays along the third axis (depth).
- **np.split(a, indices_or_sections)**: Split an array into multiple sub-arrays.
- **np.array_split(a, indices_or_sections)**: Split an array into multiple sub-arrays, allowing unequal sizes.
- **np.tile(a, reps)**: Construct an array by repeating a smaller array multiple times.
- **np.repeat(a, repeats, axis=None)**: Repeat elements of an array.

## Array Creation Functions
- **np.array()**: Create an array.
- **np.zeros()**: Create an array filled with zeros.
- **np.ones()**: Create an array filled with ones.
- **np.empty()**: Create an uninitialized array.
- **np.full()**: Create an array filled with a specified value.
- **np.eye()**: Create a 2D array with ones on the diagonal and zeros elsewhere.
- **np.identity()**: Create an identity matrix.
- **np.arange()**: Create an array with a range of values.
- **np.linspace()**: Create an array with linearly spaced values.
- **np.random.rand()**: Create an array with random values from a uniform distribution.
- **np.random.randn()**: Create an array with random values from a normal distribution.
- **np.fromiter()**: Create an array from an iterable.
- **np.frombuffer()**: Create an array from a buffer.
- **np.asarray()**: Convert input to an array.
- **np.copy()**: Create a copy of an array.

## Basic Array Operations
- **np.add()**: Add arrays element-wise.
- **np.subtract()**: Subtract arrays element-wise.
- **np.multiply()**: Multiply arrays element-wise.
- **np.divide()**: Divide arrays element-wise.
- **np.floor_divide()**: Perform floor division element-wise.
- **np.mod()**: Compute the modulus element-wise.
- **np.remainder()**: Compute the remainder element-wise.
- **np.power()**: Raise elements to a power element-wise.
- **np.float_power()**: Raise elements to a power element-wise with floating-point precision.
- **np.fmod()**: Compute the element-wise remainder of division.
- **np.negative()**: Compute the negative of each element.
- **np.sign()**: Compute the sign of each element.
- **np.reciprocal()**: Compute the reciprocal of each element.

## Array Aggregation Functions
- **np.sum()**: Compute the sum of array elements.
- **np.prod()**: Compute the product of array elements.
- **np.cumsum()**: Compute the cumulative sum of array elements.
- **np.cumprod()**: Compute the cumulative product of array elements.
- **np.min()**: Find the minimum value in an array.
- **np.max()**: Find the maximum value in an array.
- **np.argmin()**: Find the index of the minimum value in an array.
- **np.argmax()**: Find the index of the maximum value in an array.
- **np.mean()**: Compute the mean of array elements.
- **np.std()**: Compute the standard deviation of array elements.
- **np.var()**: Compute the variance of array elements.
- **np.median()**: Compute the median of array elements.
- **np.percentile()**: Compute the percentile of array elements.

## Statistical Operations
- **np.histogram()**: Compute the histogram of array elements.
- **np.corrcoef()**: Compute the correlation coefficient matrix.
- **np.cov()**: Compute the covariance matrix.
- **np.diff()**: Compute the n-th discrete difference along a given axis.
- **np.gradient()**: Compute the gradient of an array.
- **np.quantile()**: Compute the quantile of array elements.

## Manipulation and Indexing Functions
- **np.take()**: Take elements from an array along an axis.
- **np.put()**: Replaces specified elements of an array with given values.
- **np.choose()**: Construct an array from an index array and a set of arrays to choose from.
- **np.where()**: Return elements chosen from `x` or `y` depending on condition.
- **np.nonzero()**: Return the indices of the elements that are non-zero.
- **np.extract()**: Return the elements of an array that satisfy some condition.
- **np.putmask()**: Changes elements of an array based on a condition.
- **np.delete()**: Return a new array with sub-arrays along an axis deleted.
- **np.insert()**: Insert values along the given axis before the given indices.

## Array Properties
- **.shape**: Tuple of array dimensions.
- **.ndim**: Number of array dimensions.
- **.size**: Number of elements in the array.
- **.dtype**: Data type of the array elements.
- **.itemsize**: Length of one array element in bytes.
- **.T**: Transpose of the array.
- **.flatten()**: Return a copy of the array collapsed into one dimension.
- **.reshape()**: Gives a new shape to an array without changing its data.
- **.ravel()**: Return a contiguous flattened array.

## Matrix Operations
- **np.dot()**: Dot product of two arrays.
- **np.vdot()**: Return the dot product of two vectors.
- **np.cross()**: Return the cross product of two vectors.
- **np.inner()**: Return the inner product of two arrays.
- **np.outer()**: Compute the outer product of two vectors.
- **np.matmul()**: Matrix product of two arrays.
- **np.linalg.inv()**: Compute the (multiplicative) inverse of a matrix.
- **np.linalg.det()**: Compute the determinant of an array.
- **np.linalg.eig()**: Compute the eigenvalues and right eigenvectors of a square array.

## Matrix Functions
- **np.trace()**: Return the sum along diagonals of the array.
- **np.eye()**: Return a 2-D array with ones on the diagonal and zeros elsewhere.
- **np.identity()**: Return the identity array.
- **np.diag()**: Extract a diagonal or construct a diagonal array.
- **np.kron()**: Compute the Kronecker product, a composite array made of blocks of the second array scaled by the first.
