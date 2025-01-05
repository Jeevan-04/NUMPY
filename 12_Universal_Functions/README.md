# Numpy Universal Functions

This document provides an overview of various Numpy universal functions used in the `lesson.py` file.

## Table of Contents
1. [Mathematical Operations](#mathematical-operations)
2. [Comparison Operations](#comparison-operations)
3. [Logical Operations](#logical-operations)
4. [Trigonometric Functions](#trigonometric-functions)
5. [Hyperbolic Functions](#hyperbolic-functions)
6. [Exponential and Logarithmic Functions](#exponential-and-logarithmic-functions)
7. [Rounding Functions](#rounding-functions)
8. [Statistical Functions](#statistical-functions)
9. [Bitwise Operations](#bitwise-operations)
10. [Special Functions](#special-functions)
11. [Custom Universal Functions](#custom-universal-functions)

## Mathematical Operations
- `np.add()`: Adds two arrays element-wise.
- `np.subtract()`: Subtracts the second array from the first element-wise.
- `np.multiply()`: Multiplies two arrays element-wise.
- `np.divide()`: Divides the first array by the second element-wise.
- `np.floor_divide()`: Performs floor division on two arrays element-wise.
- `np.power()`: Raises elements of the first array to the powers from the second array element-wise.
- `np.float_power()`: Raises elements of the first array to the powers from the second array element-wise, using floating-point arithmetic.
- `np.mod()`: Computes the modulus (remainder) of the first array divided by the second element-wise.
- `np.remainder()`: Computes the remainder of the division of the first array by the second element-wise.
- `np.fmod()`: Computes the element-wise remainder of division.
- `np.negative()`: Computes the numerical negative of each element in the array.
- `np.reciprocal()`: Computes the reciprocal of each element in the array.

## Comparison Operations
- `np.greater()`: Returns a boolean array where the elements are True if the first array is greater than the second element-wise.
- `np.greater_equal()`: Returns a boolean array where the elements are True if the first array is greater than or equal to the second element-wise.
- `np.less()`: Returns a boolean array where the elements are True if the first array is less than the second element-wise.
- `np.less_equal()`: Returns a boolean array where the elements are True if the first array is less than or equal to the second element-wise.
- `np.equal()`: Returns a boolean array where the elements are True if the first array is equal to the second element-wise.
- `np.not_equal()`: Returns a boolean array where the elements are True if the first array is not equal to the second element-wise.
- `np.maximum()`: Compares two arrays element-wise and returns the maximum values.
- `np.minimum()`: Compares two arrays element-wise and returns the minimum values.
- `np.fmax()`: Element-wise maximum of array elements, ignoring NaNs.
- `np.fmin()`: Element-wise minimum of array elements, ignoring NaNs.

## Logical Operations
- `np.logical_and()`: Computes the element-wise logical AND of two arrays.
- `np.logical_or()`: Computes the element-wise logical OR of two arrays.
- `np.logical_xor()`: Computes the element-wise logical XOR of two arrays.
- `np.logical_not()`: Computes the element-wise logical NOT of an array.

## Trigonometric Functions
- `np.sin()`: Computes the trigonometric sine of each element in the array.
- `np.cos()`: Computes the trigonometric cosine of each element in the array.
- `np.tan()`: Computes the trigonometric tangent of each element in the array.
- `np.arcsin()`: Computes the inverse sine of each element in the array.
- `np.arccos()`: Computes the inverse cosine of each element in the array.
- `np.arctan()`: Computes the inverse tangent of each element in the array.
- `np.arctan2()`: Computes the element-wise arctangent of the quotient of two arrays.
- `np.hypot()`: Computes the hypotenuse of the right triangle given its legs.
- `np.degrees()`: Converts angles from radians to degrees.
- `np.radians()`: Converts angles from degrees to radians.
- `np.unwrap()`: Unwraps the phase of an array by changing deltas between values to 2π complement.

## Hyperbolic Functions
- `np.sinh()`: Computes the hyperbolic sine of each element in the array.
- `np.cosh()`: Computes the hyperbolic cosine of each element in the array.
- `np.tanh()`: Computes the hyperbolic tangent of each element in the array.
- `np.arcsinh()`: Computes the inverse hyperbolic sine of each element in the array.
- `np.arccosh()`: Computes the inverse hyperbolic cosine of each element in the array.
- `np.arctanh()`: Computes the inverse hyperbolic tangent of each element in the array.

## Exponential and Logarithmic Functions
- `np.exp()`: Computes the exponential of each element in the array.
- `np.exp2()`: Computes the base-2 exponential of each element in the array.
- `np.expm1()`: Computes the exponential of each element in the array minus one.
- `np.log()`: Computes the natural logarithm of each element in the array.
- `np.log10()`: Computes the base-10 logarithm of each element in the array.
- `np.log2()`: Computes the base-2 logarithm of each element in the array.
- `np.log1p()`: Computes the natural logarithm of one plus each element in the array.
- `np.logaddexp()`: Computes the logarithm of the sum of exponentiations of the inputs.
- `np.logaddexp2()`: Computes the base-2 logarithm of the sum of exponentiations of the inputs.

## Rounding Functions
- `np.ceil()`: Rounds each element in the array to the nearest integer greater than or equal to that element.
- `np.floor()`: Rounds each element in the array to the nearest integer less than or equal to that element.
- `np.rint()`: Rounds each element in the array to the nearest integer.
- `np.round()`: Rounds each element in the array to the nearest integer.
- `np.fix()`: Rounds each element in the array towards zero.
- `np.trunc()`: Truncates each element in the array to the nearest integer towards zero.

## Statistical Functions
- `np.sum()`: Computes the sum of array elements.
- `np.prod()`: Computes the product of array elements.
- `np.mean()`: Computes the arithmetic mean of array elements.
- `np.var()`: Computes the variance of array elements.
- `np.std()`: Computes the standard deviation of array elements.
- `np.min()`: Finds the minimum value in an array.
- `np.max()`: Finds the maximum value in an array.
- `np.median()`: Computes the median of array elements.
- `np.percentile()`: Computes the nth percentile of array elements.
- `np.quantile()`: Computes the nth quantile of array elements.

## Bitwise Operations
- `np.bitwise_and()`: Computes the bitwise AND of two arrays element-wise.
- `np.bitwise_or()`: Computes the bitwise OR of two arrays element-wise.
- `np.bitwise_xor()`: Computes the bitwise XOR of two arrays element-wise.
- `np.invert()`: Computes the bitwise NOT of an array.
- `np.left_shift()`: Shifts the bits of an array to the left.
- `np.right_shift()`: Shifts the bits of an array to the right.

## Special Functions
- `np.abs()`: Computes the absolute value of each element in the array.
- `np.copysign()`: Copies the sign of the second array to the first array element-wise.
- `np.sign()`: Returns an element-wise indication of the sign of a number.
- `np.spacing()`: Returns the distance between a number and the nearest adjacent number.
- `np.sinc()`: Computes the sinc function of each element in the array.
- `np.i0()`: Computes the modified Bessel function of the first kind of order 0.

## Custom Universal Functions
- `np.vectorize()`: Vectorizes a Python function so it can be applied element-wise to arrays.
- `np.frompyfunc()`: Takes an arbitrary Python function and returns a Numpy ufunc.
