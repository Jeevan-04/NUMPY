# Numpy Indexing and Masking Functions

## Table of Contents
1. [Introduction](#introduction)
2. [Basic Indexing](#basic-indexing)
   - [np.take()](#nptake)
   - [np.put()](#npput)
   - [np.choose()](#npchoose)
3. [Advanced Indexing](#advanced-indexing)
   - [np.ix_()](#npix_)
   - [np.r_[]](#npr_)
   - [np.c_[]](#npc_)
4. [Boolean Indexing](#boolean-indexing)
   - [np.where()](#npwhere)
   - [np.nonzero()](#npnonzero)
   - [np.extract()](#npextract)
5. [Masking](#masking)
   - [np.putmask()](#npputmask)
   - [np.ma.masked_where()](#npma-masked_where)
   - [np.ma.masked_equal()](#npma-masked_equal)
   - [np.ma.masked_greater()](#npma-masked_greater)
   - [np.ma.masked_less()](#npma-masked_less)
   - [np.ma.masked_inside()](#npma-masked_inside)
   - [np.ma.masked_outside()](#npma-masked_outside)
6. [Fancy Indexing](#fancy-indexing)
   - [np.ogrid[]](#npogrid)
   - [np.mgrid[]](#npmgrid)
7. [Special Indexing](#special-indexing)
   - [np.flat](#npflat)
   - [np.ndindex()](#npndindex)
   - [np.ndenumerate()](#npndenumerate)
8. [Index Modification](#index-modification)
   - [np.delete()](#npdelete)
   - [np.insert()](#npinsert)
   - [np.append()](#npappend)
9. [Slicing Utilities](#slicing-utilities)
   - [np.s_[]](#nps_)
   - [np.index_exp[]](#npindex_exp)
   - [np.take_along_axis()](#nptake_along_axis)
   - [np.put_along_axis()](#npput_along_axis)
   - [np.unravel_index()](#npunravel_index)
   - [np.ravel_multi_index()](#npravel_multi_index)
   - [np.isin()](#npisin)
   - [np.ma.getmask()](#npma-getmask)
   - [np.ma.mask_or()](#npma-mask_or)
   - [np.ma.mask_and()](#npma-mask_and)
   - [np.ma.nomask](#npma-nomask)
   - [np.diag_indices()](#npdiag_indices)
   - [np.triu_indices()](#nptriu_indices)
   - [np.tril_indices()](#nptril_indices)
   - [np.lib.stride_tricks.as_strided()](#nplib-stride_tricksas_strided)
   - [np.meshgrid()](#npmeshgrid)
   - [np.broadcast_to()](#npbroadcast_to)
   - [np.squeeze()](#npsqueeze)
   - [np.roll()](#nproll)
   - [np.rollaxis()](#nprollaxis)
   - [np.swapaxes()](#npswapaxes)
   - [np.moveaxis()](#npmoveaxis)
   - [np.transpose()](#nptranspose)
   - [np.arange()](#nparange)
   - [np.linspace()](#nplinspace)
   - [np.logspace()](#nplogspace)
   - [np.eye()](#npeye)
   - [np.identity()](#npidentity)
   - [np.diag()](#npdiag)

## Introduction
This document provides an overview of various Numpy indexing and masking functions, along with examples and descriptions.

## Basic Indexing

### np.take()
Select elements from an array along an axis using indices.
```python
a = np.array([4, 3, 5, 7, 6, 8])
indices = [0, 1, 4]
np.take(a, indices)  # [4 3 6]
```

### np.put()
Replaces specified elements of an array with given values.
```python
np.put(a, [0, 2], [-44, -55])
a  # [-44 3 -55 7 6 8]
```

### np.choose()
Constructs an array from an index array and a set of arrays to choose from.
```python
choices = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
np.choose([2, 0, 1, 2], choices)  # [ 9  2  7 12]
```

## Advanced Indexing

### np.ix_()
Construct an open mesh from multiple sequences.
```python
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
np.ix_(a, b)
```

### np.r_[]
Translates slice objects to concatenation along the first axis.
```python
np.r_[1:4, 0, 4]  # [1 2 3 0 4]
```

### np.c_[]
Translates slice objects to concatenation along the second axis.
```python
np.c_[1:4, 0:3]  # [[1 0] [2 1] [3 2]]
```

## Boolean Indexing

### np.where()
Return elements chosen from `x` or `y` depending on `condition`.
```python
a = np.array([1, 2, 3, 4])
np.where(a > 2)  # (array([2, 3]),)
```

### np.nonzero()
Return the indices of the elements that are non-zero.
```python
np.nonzero(a > 2)  # (array([2, 3]),)
```

### np.extract()
Return the elements of an array that satisfy some condition.
```python
np.extract(a > 2, a)  # [3 4]
```

## Masking

### np.putmask()
Changes elements of an array based on conditional and input values.
```python
np.putmask(a, a > 2, [42, 43, 44, 45])
a  # [ 1  2 42 43]
```

### np.ma.masked_where()
Mask an array where a condition is met.
```python
np.ma.masked_where(a > 2, a)  # [1 2 -- --]
```

### np.ma.masked_equal()
Mask an array where equal to a given value.
```python
np.ma.masked_equal(a, 2)  # [1 -- 3 4]
```

### np.ma.masked_greater()
Mask an array where greater than a given value.
```python
np.ma.masked_greater(a, 2)  # [1 2 -- --]
```

### np.ma.masked_less()
Mask an array where less than a given value.
```python
np.ma.masked_less(a, 2)  # [-- 2 3 4]
```

### np.ma.masked_inside()
Mask an array where inside a given interval.
```python
np.ma.masked_inside(a, 2, 3)  # [1 -- -- 4]
```

### np.ma.masked_outside()
Mask an array where outside a given interval.
```python
np.ma.masked_outside(a, 2, 3)  # [-- 2 3 --]
```

## Fancy Indexing

### np.ogrid[]
Open mesh grid.
```python
np.ogrid[0:5, 0:5]
```

### np.mgrid[]
Dense mesh grid.
```python
np.mgrid[0:5, 0:5]
```

## Special Indexing

### np.flat
A 1-D iterator over the array.
```python
a = np.array([[1, 2], [3, 4]])
a.flat[1]  # 2
```

### np.ndindex()
An iterator over the N-dimensional indices.
```python
list(np.ndindex(a.shape))  # [(0, 0), (0, 1), (1, 0), (1, 1)]
```

### np.ndenumerate()
An iterator yielding pairs of array coordinates and values.
```python
list(np.ndenumerate(a))  # [((0, 0), 1), ((0, 1), 2), ((1, 0), 3), ((1, 1), 4)]
```

## Index Modification

### np.delete()
Return a new array with sub-arrays along an axis deleted.
```python
a = np.array([1, 2, 3, 4])
np.delete(a, [1, 3])  # [1 3]
```

### np.insert()
Insert values along the given axis before the given indices.
```python
np.insert(a, 1, 5)  # [1 5 2 3 4]
```

### np.append()
Append values to the end of an array.
```python
np.append(a, [5, 6])  # [1 2 3 4 5 6]
```

## Slicing Utilities

### np.s_[]
Return a slice object.
```python
a[np.s_[1:3]]  # [2 3]
```

### np.index_exp[]
Return an index expression.
```python
a[np.index_exp[1:3]]  # [2 3]
```

### np.take_along_axis()
Take values from the input array by matching 1d index and data slices.
```python
a = np.array([[10, 30, 20], [60, 40, 50]])
indices = np.array([[0, 1, 2], [2, 0, 1]])
np.take_along_axis(a, indices, axis=1)  # [[10 30 20] [50 60 40]]
```

### np.put_along_axis()
Put values into the destination array by matching 1d index and data slices.
```python
np.put_along_axis(a, indices, [1, 2, 3, 4, 5, 6], axis=1)
a  # [[ 1  2  3] [ 4  5  6]]
```

### np.unravel_index()
Converts a flat index or array of flat indices into a tuple of coordinate arrays.
```python
np.unravel_index([22, 41, 37], (7, 6))  # (array([3, 6, 6]), array([4, 5, 1]))
```

### np.ravel_multi_index()
Converts a tuple of coordinate arrays to flat indices.
```python
np.ravel_multi_index(([3, 6, 6], [4, 5, 1]), (7, 6))  # [22 41 37]
```

### np.isin()
Returns a boolean array where elements of `a` are in `test_elements`.
```python
np.isin(a, [2, 3])  # [False  True  True False]
```

### np.ma.getmask()
Returns the mask of a masked array.
```python
a = np.ma.array([1, 2, 3], mask=[0, 1, 0])
np.ma.getmask(a)  # [False  True False]
```

### np.ma.mask_or()
Combines two masks with the logical OR operation.
```python
b = np.ma.array([4, 5, 6], mask=[1, 0, 0])
np.ma.mask_or(a, b)  # [ True  True False]
```

### np.ma.mask_and()
Combines two masks with the logical AND operation.
```python
np.ma.mask_and(a, b)  # [False False False]
```

### np.ma.nomask
A singleton mask indicating no masking.
```python
np.ma.nomask  # False
```

### np.diag_indices()
Return the indices to access the main diagonal of an array.
```python
np.diag_indices(3)  # (array([0, 1, 2]), array([0, 1, 2]))
```

### np.triu_indices()
Return the indices for the upper-triangle of an array.
```python
np.triu_indices(3)  # (array([0, 0, 0, 1, 1, 2]), array([0, 1, 2, 1, 2, 2]))
```

### np.tril_indices()
Return the indices for the lower-triangle of an array.
```python
np.tril_indices(3)  # (array([0, 1, 1, 2, 2, 2]), array([0, 0, 1, 0, 1, 2]))
```

### np.lib.stride_tricks.as_strided()
Create a view into the array with the given shape and strides.
```python
a = np.array([1, 2, 3, 4, 5, 6])
np.lib.stride_tricks.as_strided(a, shape=(3, 2), strides=(8, 8))
```

### np.meshgrid()
Return coordinate matrices from coordinate vectors.
```python
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
np.meshgrid(x, y)
```

### np.broadcast_to()
Broadcast an array to a new shape.
```python
a = np.array([1, 2, 3])
np.broadcast_to(a, (3, 3))
```

### np.squeeze()
Remove single-dimensional entries from the shape of an array.
```python
a = np.array([[[0], [1], [2]]])
np.squeeze(a)  # [0 1 2]
```

### np.roll()
Roll array elements along a given axis.
```python
a = np.array([1, 2, 3, 4, 5])
np.roll(a, 2)  # [4 5 1 2 3]
```

### np.rollaxis()
Roll the specified axis backwards.
```python
a = np.ones((3, 4, 5))
np.rollaxis(a, 2, 0).shape  # (5, 3, 4)
```

### np.swapaxes()
Interchange two axes of an array.
```python
a = np.array([[1, 2, 3]])
np.swapaxes(a, 0, 1)  # [[1] [2] [3]]
```

### np.moveaxis()
Move axes of an array to new positions.
```python
a = np.ones((3, 4, 5))
np.moveaxis(a, 0, -1).shape  # (4, 5, 3)
```

### np.transpose()
Permute the dimensions of an array.
```python
a = np.array([[1, 2, 3]])
np.transpose(a)  # [[1] [2] [3]]
```

### np.arange()
Return evenly spaced values within a given interval.
```python
np.arange(3)  # [0 1 2]
```

### np.linspace()
Return evenly spaced numbers over a specified interval.
```python
np.linspace(2.0, 3.0, num=5)  # [2.   2.25 2.5  2.75 3.  ]
```

### np.logspace()
Return numbers spaced evenly on a log scale.
```python
np.logspace(2.0, 3.0, num=4)  # [ 100.  215.443469  464.158883 1000. ]
```

### np.eye()
Return a 2-D array with ones on the diagonal and zeros elsewhere.
```python
np.eye(3)  # [[1. 0. 0.] [0. 1. 0.] [0. 0. 1.]]
```

### np.identity()
Return the identity array.
```python
np.identity(3)  # [[1. 0. 0.] [0. 1. 0.] [0. 0. 1.]]
```

### np.diag()
Extract a diagonal or construct a diagonal array.
```python
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
np.diag(a)  # [1 5 9]
```