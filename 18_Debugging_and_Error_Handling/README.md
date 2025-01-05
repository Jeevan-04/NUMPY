# NumPy Debugging and Error Handling

## Table of Contents
1. [Introduction](#introduction)
2. [Functions](#functions)
    - [np.seterr()](#npseterr)
    - [np.geterr()](#npgeterr)
    - [np.errstate()](#nperrstate)
    - [np.set_printoptions()](#npset_printoptions)
    - [np.get_printoptions()](#npget_printoptions)
    - [np.savetxt()](#npsavetxt)
    - [np.loadtxt()](#nploadtxt)
    - [np.save()](#npsave)
    - [np.load()](#npload)
    - [np.ma.set_fill_value()](#npmaset_fill_value)
    - [np.ma.masked_where()](#npma_masked_where)
    - [np.ma.masked_equal()](#npma_masked_equal)
    - [np.ma.masked_greater()](#npma_masked_greater)
    - [np.ma.masked_less()](#npma_masked_less)
    - [np.ma.masked_inside()](#npma_masked_inside)
    - [np.ma.masked_outside()](#npma_masked_outside)
    - [np.trace()](#nptrace)
    - [np.logical_and()](#nplogical_and)
    - [np.logical_or()](#nplogical_or)
    - [np.logical_xor()](#nplogical_xor)
    - [np.isin()](#npisin)
    - [np.isfinite()](#npisfinite)
    - [np.isnan()](#npisnan)
    - [np.isinf()](#npisinf)
    - [np.iscomplexobj()](#npiscomplexobj)
    - [np.isrealobj()](#npisrealobj)
    - [np.isclose()](#npisclose)
    - [np.equal()](#npequal)
    - [np.issubdtype()](#npissubdtype)

## Introduction
This document provides an overview of various NumPy functions related to debugging, error handling, and array operations.

## Functions

### np.seterr()
Sets how floating-point errors are handled.

### np.geterr()
Gets the current handling of floating-point errors.

### np.errstate()
Context manager for floating-point error handling.

### np.set_printoptions()
Sets print options for NumPy arrays.

### np.get_printoptions()
Gets the current print options for NumPy arrays.

### np.savetxt()
Saves an array to a text file.

### np.loadtxt()
Loads data from a text file into an array.

### np.save()
Saves an array to a binary file in NumPy `.npy` format.

### np.load()
Loads an array from a binary file in NumPy `.npy` format.

### np.ma.set_fill_value()
Sets the fill value of a masked array.

### np.ma.masked_where()
Masks an array where a condition is met.

### np.ma.masked_equal()
Masks an array where values are equal to a given value.

### np.ma.masked_greater()
Masks an array where values are greater than a given value.

### np.ma.masked_less()
Masks an array where values are less than a given value.

### np.ma.masked_inside()
Masks an array where values are inside a given interval.

### np.ma.masked_outside()
Masks an array where values are outside a given interval.

### np.trace()
Returns the sum along diagonals of the array.

### np.logical_and()
Computes the element-wise logical AND of two arrays.

### np.logical_or()
Computes the element-wise logical OR of two arrays.

### np.logical_xor()
Computes the element-wise logical XOR of two arrays.

### np.isin()
Checks if elements of one array are in another array.

### np.isfinite()
Checks if elements are finite (not infinity or NaN).

### np.isnan()
Checks if elements are NaN (Not a Number).

### np.isinf()
Checks if elements are infinite.

### np.iscomplexobj()
Checks if the input is a complex type or an array of complex numbers.

### np.isrealobj()
Checks if the input is a real type or an array of real numbers.

### np.isclose()
Checks if elements of two arrays are element-wise equal within a tolerance.

### np.equal()
Checks if elements of two arrays are element-wise equal.

### np.issubdtype()
Checks if a given dtype is a subdtype of another dtype.
