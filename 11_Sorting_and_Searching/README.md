# NumPy Sorting and Searching

## Table of Contents
1. [Introduction](#introduction)
2. [Sorting Functions](#sorting-functions)
    - [np.sort()](#npsort)
    - [np.argsort()](#npargsort)
    - [np.lexsort()](#nplexsort)
    - [np.partition()](#nppartition)
    - [np.argpartition()](#npargpartition)
3. [Searching Functions](#searching-functions)
    - [np.searchsorted()](#npsearchsorted)
    - [np.where()](#npwhere)
    - [np.nonzero()](#npnonzero)
    - [np.argmax()](#npargmax)
    - [np.argmin()](#npargmin)
    - [np.nanargmax()](#npnanargmax)
    - [np.nanargmin()](#npnanargmin)
    - [np.extract()](#npextract)
    - [np.flatnonzero()](#npflatnonzero)
4. [Set Operations](#set-operations)
    - [np.unique()](#npunique)
    - [np.in1d()](#npin1d)
    - [np.intersect1d()](#npintersect1d)
    - [np.setdiff1d()](#npsetdiff1d)
    - [np.setxor1d()](#npsetxor1d)
    - [np.union1d()](#npunion1d)

## Introduction
This document provides examples and descriptions of various sorting, searching, and set operations available in NumPy.

## Sorting Functions

### np.sort()
Sorts an array.
```python
arr = np.array([3, 1, 2, 5, 4])
sorted_arr = np.sort(arr)
```

### np.argsort()
Returns the indices that would sort an array.
```python
argsorted_indices = np.argsort(arr)
```

### np.lexsort()
Performs an indirect stable sort using a sequence of keys.
```python
names = ('John', 'Jane', 'Doe', 'Alice')
ages = (25, 30, 20, 22)
lexsorted_indices = np.lexsort((ages, names))
```

### np.partition()
Partitions an array into two parts.
```python
partitioned_arr = np.partition(arr, 3)
```

### np.argpartition()
Returns the indices that would partition an array.
```python
argpartitioned_indices = np.argpartition(arr, 3)
```

## Searching Functions

### np.searchsorted()
Finds indices where elements should be inserted to maintain order.
```python
searchsorted_index = np.searchsorted(arr, 3)
```

### np.where()
Returns elements chosen from `x` or `y` depending on `condition`.
```python
where_indices = np.where(arr > 3)
```

### np.nonzero()
Returns the indices of the elements that are non-zero.
```python
nonzero_indices = np.nonzero(arr)
```

### np.argmax()
Returns the indices of the maximum values along an axis.
```python
argmax_index = np.argmax(arr)
```

### np.argmin()
Returns the indices of the minimum values along an axis.
```python
argmin_index = np.argmin(arr)
```

### np.nanargmax()
Returns the indices of the maximum values in an array, ignoring NaNs.
```python
nanargmax_index = np.nanargmax(arr_with_nan)
```

### np.nanargmin()
Returns the indices of the minimum values in an array, ignoring NaNs.
```python
nanargmin_index = np.nanargmin(arr_with_nan)
```

### np.extract()
Returns the elements of an array that satisfy some condition.
```python
condition = arr > 2
extracted_elements = np.extract(condition, arr)
```

### np.flatnonzero()
Returns indices that are non-zero in the flattened version of the array.
```python
flatnonzero_indices = np.flatnonzero(arr)
```

## Set Operations

### np.unique()
Finds the unique elements of an array.
```python
unique_elements = np.unique(arr1)
```

### np.in1d()
Tests whether each element of an array is also present in a second array.
```python
in1d_result = np.in1d(arr1, arr2)
```

### np.intersect1d()
Finds the intersection of two arrays.
```python
intersect1d_result = np.intersect1d(arr1, arr2)
```

### np.setdiff1d()
Finds the set difference of two arrays.
```python
setdiff1d_result = np.setdiff1d(arr1, arr2)
```

### np.setxor1d()
Finds the set exclusive-or of two arrays.
```python
setxor1d_result = np.setxor1d(arr1, arr2)
```

### np.union1d()
Finds the union of two arrays.
```python
union1d_result = np.union1d(arr1, arr2)
