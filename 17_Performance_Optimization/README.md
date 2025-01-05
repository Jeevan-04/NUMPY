# Performance Optimization with NumPy

## Table of Contents
1. [Introduction](#introduction)
2. [Functions Used](#functions-used)
    - [np.set_printoptions](#np-set_printoptions)
    - [np.setbufsize](#np-setbufsize)
    - [np.seterr](#np-seterr)
    - [np.geterr](#np-geterr)
    - [np.get_printoptions](#np-get_printoptions)
    - [np.memmap](#np-memmap)
    - [np.load](#np-load)
    - [np.einsum](#np-einsum)
    - [np.vectorize](#np-vectorize)
    - [np.interp](#np-interp)
    - [np.fromiter](#np-fromiter)
    - [np.bincount](#np-bincount)
    - [np.histogram](#np-histogram)
    - [np.matmul](#np-matmul)
    - [np.dot](#np-dot)
    - [np.linalg.multi_dot](#np-linalg-multi_dot)
    - [np.nan_to_num](#np-nan_to_num)
    - [np.clip](#np-clip)
    - [np.tile](#np-tile)
    - [np.broadcast_to](#np-broadcast_to)
    - [np.arange](#np-arange)
    - [np.ones](#np-ones)
    - [np.zeros](#np-zeros)
    - [np.empty](#np-empty)
    - [np.zeros_like](#np-zeros_like)
    - [np.ones_like](#np-ones_like)
    - [np.full](#np-full)

## Introduction
This project demonstrates various performance optimization techniques using NumPy. The script includes examples of setting print options, handling errors, memory mapping, and using efficient NumPy functions.

## Functions Used

### np.set_printoptions
Sets the print options for NumPy arrays.

### np.setbufsize
Sets the size of the buffer used in ufuncs.

### np.seterr
Sets how floating-point errors are handled.

### np.geterr
Gets the current error handling settings.

### np.get_printoptions
Gets the current print options.

### np.memmap
Creates a memory-mapped array.

### np.load
Loads arrays or pickled objects from .npy, .npz, or pickled files.

### np.einsum
Evaluates the Einstein summation convention on the operands.

### np.vectorize
Vectorizes a function, making it applicable to arrays.

### np.interp
Performs one-dimensional linear interpolation.

### np.fromiter
Creates a new 1-dimensional array from an iterable object.

### np.bincount
Counts the number of occurrences of each value in an array of non-negative integers.

### np.histogram
Computes the histogram of a dataset.

### np.matmul
Performs matrix multiplication.

### np.dot
Computes the dot product of two arrays.

### np.linalg.multi_dot
Computes the dot product of two or more arrays in a single function call.

### np.nan_to_num
Replaces NaNs with zero and infinities with large finite numbers.

### np.clip
Clips (limits) the values in an array.

### np.tile
Constructs an array by repeating A the number of times given by reps.

### np.broadcast_to
Broadcasts an array to a new shape.

### np.arange
Returns evenly spaced values within a given interval.

### np.ones
Returns a new array of given shape and type, filled with ones.

### np.zeros
Returns a new array of given shape and type, filled with zeros.

### np.empty
Returns a new array of given shape and type, without initializing entries.

### np.zeros_like
Returns an array of zeros with the same shape and type as a given array.

### np.ones_like
Returns an array of ones with the same shape and type as a given array.

### np.full
Returns a new array of given shape and type, filled with a specified value.
