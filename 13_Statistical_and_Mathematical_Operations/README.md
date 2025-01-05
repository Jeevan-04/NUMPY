# NumPy Statistical and Mathematical Operations

## Table of Contents
1. [Introduction](#introduction)
2. [Statistical Functions](#statistical-functions)
3. [Mathematical Functions](#mathematical-functions)
    - [Basic Arithmetic](#basic-arithmetic)
    - [Trigonometric Functions](#trigonometric-functions)
    - [Hyperbolic Functions](#hyperbolic-functions)
    - [Exponential and Logarithmic Functions](#exponential-and-logarithmic-functions)
    - [Rounding](#rounding)
    - [Advanced Math Functions](#advanced-math-functions)
    - [Specialized Operations](#specialized-operations)
    - [Matrix-Specific](#matrix-specific)

## Introduction
This document provides an overview of various statistical and mathematical functions available in NumPy, along with examples of their usage.

## Statistical Functions
- **np.mean()**: Calculate the mean of an array.
- **np.median()**: Calculate the median of an array.
- **np.average()**: Calculate the weighted average of an array.
- **np.std()**: Calculate the standard deviation of an array.
- **np.var()**: Calculate the variance of an array.
- **np.ptp()**: Calculate the peak-to-peak (range) of an array.
- **np.min()**: Find the minimum value in an array.
- **np.max()**: Find the maximum value in an array.
- **np.percentile()**: Calculate the nth percentile of an array.
- **np.quantile()**: Calculate the nth quantile of an array.
- **np.corrcoef()**: Calculate the correlation coefficient matrix.
- **np.cov()**: Calculate the covariance matrix.
- **np.histogram()**: Compute the histogram of an array.
- **np.histogram_bin_edges()**: Compute the bin edges for a histogram.
- **np.bincount()**: Count the number of occurrences of each value in an array.
- **np.digitize()**: Return the indices of the bins to which each value in input array belongs.
- **np.nanmean()**: Calculate the mean of an array, ignoring NaNs.
- **np.nanmedian()**: Calculate the median of an array, ignoring NaNs.
- **np.nanstd()**: Calculate the standard deviation of an array, ignoring NaNs.
- **np.nanvar()**: Calculate the variance of an array, ignoring NaNs.
- **np.nanmin()**: Find the minimum value in an array, ignoring NaNs.
- **np.nanmax()**: Find the maximum value in an array, ignoring NaNs.
- **np.nansum()**: Calculate the sum of an array, ignoring NaNs.
- **np.nanpercentile()**: Calculate the nth percentile of an array, ignoring NaNs.
- **np.nanquantile()**: Calculate the nth quantile of an array, ignoring NaNs.

## Mathematical Functions

### Basic Arithmetic
- **np.add()**: Add two arrays element-wise.
- **np.subtract()**: Subtract one array from another element-wise.
- **np.multiply()**: Multiply two arrays element-wise.
- **np.divide()**: Divide one array by another element-wise.
- **np.floor_divide()**: Perform floor division on two arrays element-wise.
- **np.mod()**: Calculate the modulus of two arrays element-wise.
- **np.power()**: Raise one array to the power of another element-wise.
- **np.float_power()**: Raise one array to the power of another element-wise, using floating-point arithmetic.
- **np.remainder()**: Calculate the remainder of division of two arrays element-wise.

### Trigonometric Functions
- **np.sin()**: Calculate the sine of each element in an array.
- **np.cos()**: Calculate the cosine of each element in an array.
- **np.tan()**: Calculate the tangent of each element in an array.
- **np.arcsin()**: Calculate the inverse sine of each element in an array.
- **np.arccos()**: Calculate the inverse cosine of each element in an array.
- **np.arctan()**: Calculate the inverse tangent of each element in an array.
- **np.arctan2()**: Calculate the inverse tangent of the quotient of two arrays.
- **np.hypot()**: Calculate the hypotenuse of a right triangle given its legs.
- **np.degrees()**: Convert angles from radians to degrees.
- **np.radians()**: Convert angles from degrees to radians.
- **np.unwrap()**: Unwrap by changing deltas between values to 2π complement.

### Hyperbolic Functions
- **np.sinh()**: Calculate the hyperbolic sine of each element in an array.
- **np.cosh()**: Calculate the hyperbolic cosine of each element in an array.
- **np.tanh()**: Calculate the hyperbolic tangent of each element in an array.
- **np.arcsinh()**: Calculate the inverse hyperbolic sine of each element in an array.
- **np.arccosh()**: Calculate the inverse hyperbolic cosine of each element in an array.
- **np.arctanh()**: Calculate the inverse hyperbolic tangent of each element in an array.

### Exponential and Logarithmic Functions
- **np.exp()**: Calculate the exponential of each element in an array.
- **np.exp2()**: Calculate 2 raised to the power of each element in an array.
- **np.expm1()**: Calculate exp(x) - 1 for each element in an array.
- **np.log()**: Calculate the natural logarithm of each element in an array.
- **np.log10()**: Calculate the base-10 logarithm of each element in an array.
- **np.log2()**: Calculate the base-2 logarithm of each element in an array.
- **np.log1p()**: Calculate log(1 + x) for each element in an array.
- **np.logaddexp()**: Calculate the logarithm of the sum of exponentiations of the inputs.
- **np.logaddexp2()**: Calculate the base-2 logarithm of the sum of exponentiations of the inputs.

### Rounding
- **np.ceil()**: Return the ceiling of each element in an array.
- **np.floor()**: Return the floor of each element in an array.
- **np.rint()**: Round elements of an array to the nearest integer.
- **np.round()**: Round elements of an array to the nearest integer.
- **np.fix()**: Round elements of an array towards zero.
- **np.trunc()**: Truncate elements of an array to the nearest integer towards zero.

### Advanced Math Functions
- **np.abs()**: Calculate the absolute value of each element in an array.
- **np.copysign()**: Change the sign of each element in an array to match the sign of another array.
- **np.sign()**: Extract the sign of each element in an array.
- **np.spacing()**: Calculate the distance between a number and the nearest adjacent number.
- **np.sinc()**: Calculate the sinc function of each element in an array.
- **np.i0()**: Calculate the modified Bessel function of the first kind of order 0 for each element in an array.
- **np.clip()**: Clip (limit) the values in an array.

### Specialized Operations
- **np.prod()**: Calculate the product of elements in an array.
- **np.sum()**: Calculate the sum of elements in an array.
- **np.cumsum()**: Calculate the cumulative sum of elements in an array.
- **np.cumprod()**: Calculate the cumulative product of elements in an array.
- **np.diff()**: Calculate the n-th discrete difference along the given axis.
- **np.ediff1d()**: Calculate the differences between consecutive elements of an array.
- **np.gradient()**: Calculate the gradient of an array.
- **np.trapz()**: Integrate along the given axis using the trapezoidal rule.

### Matrix-Specific
- **np.linalg.norm()**: Calculate the matrix or vector norm.
- **np.linalg.det()**: Calculate the determinant of a matrix.
- **np.linalg.slogdet()**: Calculate the sign and logarithm of the determinant of a matrix.
- **np.linalg.matrix_rank()**: Calculate the rank of a matrix.
- **np.linalg.multi_dot()**: Compute the dot product of two or more arrays in a single function call.
