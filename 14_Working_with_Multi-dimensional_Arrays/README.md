# Working with Multi-dimensional Arrays in NumPy

## Table of Contents
1. [Introduction](#introduction)
2. [Functions](#functions)
    - [np.reshape()](#npreshape)
    - [np.flatten()](#npflatten)
    - [np.ravel()](#npravel)
    - [np.transpose()](#nptranspose)
    - [np.swapaxes()](#npswapaxes)
    - [np.moveaxis()](#npmoveaxis)
    - [np.rollaxis()](#nprollaxis)
    - [np.expand_dims()](#npexpand_dims)
    - [np.squeeze()](#npsqueeze)
    - [np.hstack()](#nphstack)
    - [np.vstack()](#npvstack)
    - [np.dstack()](#npdstack)
    - [np.column_stack()](#npcolumn_stack)
    - [np.row_stack()](#nprow_stack)
    - [np.stack()](#npstack)
    - [np.split()](#npsplit)
    - [np.array_split()](#nparray_split)
    - [np.concatenate()](#npconcatenate)
    - [np.insert()](#npinsert)
    - [np.delete()](#npdelete)
    - [np.append()](#npappend)
    - [np.resize()](#npresize)
    - [np.unique()](#npunique)
    - [np.meshgrid()](#npmeshgrid)
    - [np.ix_()](#npix_)
    - [np.ndindex()](#npndindex)
    - [np.ndenumerate()](#npndenumerate)
    - [np.broadcast()](#npbroadcast)
    - [np.broadcast_arrays()](#npbroadcast_arrays)
    - [np.broadcast_to()](#npbroadcast_to)
    - [np.r_[]](#npr_)
    - [np.c_[]](#npc_)
    - [np.mgrid[]](#npmgrid)
    - [np.ogrid[]](#npogrid)

## Introduction
This document provides an overview of various functions available in NumPy for working with multi-dimensional arrays. Each function is described with its purpose and usage.

## Functions

### np.reshape()
Reshapes an array without changing its data.
```python
reshaped = np.reshape(arr, (3, 2))
```

### np.flatten()
Returns a copy of the array collapsed into one dimension.
```python
flattened = arr.flatten()
```

### np.ravel()
Returns a contiguous flattened array.
```python
raveled = np.ravel(arr)
```

### np.transpose()
Permutes the dimensions of an array.
```python
transposed = np.transpose(arr)
```

### np.swapaxes()
Interchanges two axes of an array.
```python
swapped = np.swapaxes(arr, 0, 1)
```

### np.moveaxis()
Moves axes of an array to new positions.
```python
moved = np.moveaxis(arr, 0, 1)
```

### np.rollaxis()
Rolls the specified axis backwards.
```python
rolled = np.rollaxis(arr, 1, 0)
```

### np.expand_dims()
Expands the shape of an array.
```python
expanded = np.expand_dims(arr, axis=0)
```

### np.squeeze()
Removes single-dimensional entries from the shape of an array.
```python
squeezed = np.squeeze(expanded)
```

### np.hstack()
Stacks arrays in sequence horizontally (column-wise).
```python
hstacked = np.hstack((arr, arr))
```

### np.vstack()
Stacks arrays in sequence vertically (row-wise).
```python
vstacked = np.vstack((arr, arr))
```

### np.dstack()
Stacks arrays in sequence depth-wise (along third dimension).
```python
dstacked = np.dstack((arr, arr))
```

### np.column_stack()
Stacks 1-D arrays as columns into a 2-D array.
```python
column_stacked = np.column_stack((arr, arr))
```

### np.row_stack()
Stacks arrays in sequence vertically (row-wise).
```python
row_stacked = np.row_stack((arr, arr))
```

### np.stack()
Joins a sequence of arrays along a new axis.
```python
stacked = np.stack((arr, arr), axis=0)
```

### np.split()
Splits an array into multiple sub-arrays.
```python
split = np.split(arr, 2)
```

### np.array_split()
Splits an array into multiple sub-arrays of equal or near-equal size.
```python
array_split = np.array_split(arr, 3)
```

### np.concatenate()
Joins a sequence of arrays along an existing axis.
```python
concatenated = np.concatenate((arr, arr), axis=0)
```

### np.insert()
Inserts values along the given axis before the given indices.
```python
inserted = np.insert(arr, 1, 99, axis=0)
```

### np.delete()
Returns a new array with sub-arrays along an axis deleted.
```python
deleted = np.delete(arr, 1, axis=0)
```

### np.append()
Appends values to the end of an array.
```python
appended = np.append(arr, [[7, 8, 9]], axis=0)
```

### np.resize()
Returns a new array with the specified shape.
```python
resized = np.resize(arr, (3, 3))
```

### np.unique()
Finds the unique elements of an array.
```python
unique = np.unique(arr)
```

### np.meshgrid()
Returns coordinate matrices from coordinate vectors.
```python
xx, yy = np.meshgrid(x, y)
```

### np.ix_()
Constructs an open mesh from multiple sequences.
```python
ix = np.ix_([0, 1], [1, 2])
```

### np.ndindex()
Generates multi-dimensional index arrays.
```python
for index in np.ndindex(arr.shape):
    print(index, arr[index])
```

### np.ndenumerate()
Returns an iterator yielding pairs of array coordinates and values.
```python
for index, value in np.ndenumerate(arr):
    print(index, value)
```

### np.broadcast()
Produces an object that mimics broadcasting.
```python
broadcasted = np.broadcast(arr, arr)
```

### np.broadcast_arrays()
Broadcasts any number of arrays against each other.
```python
broadcast_arrays = np.broadcast_arrays(arr, arr)
```

### np.broadcast_to()
Broadcasts an array to a new shape.
```python
broadcast_to = np.broadcast_to(arr, (2, 2, 3))
```

### np.r_[]
Translates slice objects to concatenation along the first axis.
```python
r_ = np.r_[arr, arr]
```

### np.c_[]
Translates slice objects to concatenation along the second axis.
```python
c_ = np.c_[arr, arr]
```

### np.mgrid[]
Returns a dense multi-dimensional "meshgrid".
```python
mgrid = np.mgrid[0:3, 0:3]
```

### np.ogrid[]
Returns an open multi-dimensional "meshgrid".
```python
ogrid = np.ogrid[0:3, 0:3]
