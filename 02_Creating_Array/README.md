# Numpy Array Creation

## Table of Contents
1. [Creating Arrays from Lists and Tuples](#creating-arrays-from-lists-and-tuples)
2. [Creating Arrays with Placeholder Content](#creating-arrays-with-placeholder-content)
3. [Creating Arrays with Ranges](#creating-arrays-with-ranges)
4. [Creating Identity Matrices](#creating-identity-matrices)
5. [Creating Arrays from Existing Data](#creating-arrays-from-existing-data)
6. [Creating Arrays with Meshgrid](#creating-arrays-with-meshgrid)

## Creating Arrays from Lists and Tuples
- `np.array(object)`: Create an array from a list or tuple.

## Creating Arrays with Placeholder Content
- `np.empty(shape)`: Create an uninitialized array.
- `np.zeros(shape)`: Create an array filled with zeros.
- `np.ones(shape)`: Create an array filled with ones.
- `np.full(shape, fill_value)`: Create an array filled with a specified value.

## Creating Arrays with Ranges
- `np.arange(start, stop, step)`: Create an array with a range of values.
- `np.linspace(start, stop, num)`: Create an array with linearly spaced values.
- `np.logspace(start, stop, num)`: Create an array with logarithmically spaced values.

## Creating Identity Matrices
- `np.eye(N)`: Create a 2-D array with ones on the diagonal and zeros elsewhere.
- `np.identity(n)`: Create a square identity matrix.

## Creating Arrays from Existing Data
- `np.asarray(a)`: Convert input to an array.
- `np.frombuffer(buffer)`: Interpret a buffer as a 1-D array.
- `np.fromfunction(function, shape)`: Construct an array by executing a function over each coordinate.
- `np.fromiter(iterable, dtype)`: Create a 1-D array from an iterable.
- `np.fromstring(string, dtype)`: Create an array from a string.

## Creating Arrays with Meshgrid
- `np.meshgrid(*xi)`: Create coordinate matrices from coordinate vectors.
