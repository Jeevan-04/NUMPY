# Numpy Arithmetic Operations

## Table of Contents
1. [Basic Arithmetic Operations](#basic-arithmetic-operations)
2. [Accumulation and Reduction](#accumulation-and-reduction)
3. [Exponential and Logarithmic Operations](#exponential-and-logarithmic-operations)
4. [Rounding Operations](#rounding-operations)
5. [Special Arithmetic Functions](#special-arithmetic-functions)
6. [Matrix-Specific Operations](#matrix-specific-operations)
7. [Element-wise Arithmetic with Broadcasting](#element-wise-arithmetic-with-broadcasting)
8. [Logical Arithmetic Operations](#logical-arithmetic-operations)
9. [Comparative Arithmetic Operations](#comparative-arithmetic-operations)
10. [Handling NaNs in Arithmetic](#handling-nans-in-arithmetic)

## Basic Arithmetic Operations
- `np.add()`: Adds two arrays element-wise.
- `np.subtract()`: Subtracts the second array from the first element-wise.
- `np.multiply()`: Multiplies two arrays element-wise.
- `np.divide()`: Divides the first array by the second element-wise.
- `np.floor_divide()`: Performs floor division on two arrays element-wise.
- `np.mod()`: Computes the modulus (remainder) of two arrays element-wise.
- `np.remainder()`: Alias for `np.mod()`.
- `np.fmod()`: Computes the element-wise remainder of division.
- `np.power()`: Raises the first array elements to the powers from the second array, element-wise.
- `np.float_power()`: Raises the first array elements to the powers from the second array, element-wise, using floating-point arithmetic.
- `np.negative()`: Computes the numerical negative of each element in the array.
- `np.abs()`: Computes the absolute value element-wise.

## Accumulation and Reduction
- `np.cumsum()`: Returns the cumulative sum of the elements along a given axis.
- `np.cumprod()`: Returns the cumulative product of the elements along a given axis.
- `np.sum()`: Returns the sum of array elements over a given axis.
- `np.prod()`: Returns the product of array elements over a given axis.
- `np.nansum()`: Returns the sum of array elements over a given axis treating NaNs as zero.
- `np.nanprod()`: Returns the product of array elements over a given axis treating NaNs as one.

## Exponential and Logarithmic Operations
- `np.exp()`: Calculates the exponential of all elements in the input array.
- `np.expm1()`: Calculates `exp(x) - 1` for all elements in the array.
- `np.exp2()`: Calculates `2**x` for all elements in the array.
- `np.log()`: Calculates the natural logarithm of all elements in the array.
- `np.log10()`: Calculates the base-10 logarithm of all elements in the array.
- `np.log2()`: Calculates the base-2 logarithm of all elements in the array.
- `np.log1p()`: Calculates `log(1 + x)` for all elements in the array.
- `np.logaddexp()`: Calculates `log(exp(x1) + exp(x2))` for all elements in the input arrays.
- `np.logaddexp2()`: Calculates `log2(2**x1 + 2**x2)` for all elements in the input arrays.

## Rounding Operations
- `np.round()`: Rounds elements of the array to the nearest integer.
- `np.around()`: Alias for `np.round()`.
- `np.rint()`: Rounds elements of the array to the nearest integer.
- `np.fix()`: Rounds elements of the array towards zero.
- `np.floor()`: Rounds elements of the array to the nearest lower integer.
- `np.ceil()`: Rounds elements of the array to the nearest higher integer.
- `np.trunc()`: Truncates elements of the array to the nearest integer towards zero.

## Special Arithmetic Functions
- `np.sqrt()`: Computes the non-negative square root of each element in the array.
- `np.cbrt()`: Computes the cube root of each element in the array.
- `np.square()`: Computes the square of each element in the array.
- `np.reciprocal()`: Computes the reciprocal of each element in the array.
- `np.sign()`: Returns an element-wise indication of the sign of a number.
- `np.clip()`: Clips (limits) the values in an array.
- `np.gradient()`: Returns the gradient of an N-dimensional array.
- `np.diff()`: Calculates the n-th discrete difference along the given axis.

## Matrix-Specific Operations
- `np.matmul()`: Matrix product of two arrays.
- `np.dot()`: Dot product of two arrays.
- `np.vdot()`: Returns the dot product of two vectors.
- `np.inner()`: Returns the inner product of two arrays.
- `np.outer()`: Returns the outer product of two arrays.
- `np.tensordot()`: Computes the tensor dot product along specified axes.
- `np.kron()`: Computes the Kronecker product of two arrays.
- `np.cross()`: Returns the cross product of two vectors.

## Element-wise Arithmetic with Broadcasting
- `np.add.outer()`: Computes the outer addition of two arrays.
- `np.subtract.outer()`: Computes the outer subtraction of two arrays.
- `np.multiply.outer()`: Computes the outer multiplication of two arrays.
- `np.divide.outer()`: Computes the outer division of two arrays.

## Logical Arithmetic Operations
- `np.bitwise_and()`: Computes the bit-wise AND of two arrays element-wise.
- `np.bitwise_or()`: Computes the bit-wise OR of two arrays element-wise.
- `np.bitwise_xor()`: Computes the bit-wise XOR of two arrays element-wise.
- `np.bitwise_not()`: Computes the bit-wise NOT of the array element-wise.
- `np.left_shift()`: Shifts the bits of an integer to the left.
- `np.right_shift()`: Shifts the bits of an integer to the right.

## Comparative Arithmetic Operations
- `np.maximum()`: Compares two arrays element-wise and returns the maximum values.
- `np.minimum()`: Compares two arrays element-wise and returns the minimum values.
- `np.fmax()`: Element-wise maximum of array elements, ignoring NaNs.
- `np.fmin()`: Element-wise minimum of array elements, ignoring NaNs.

## Handling NaNs in Arithmetic
- `np.nan_to_num()`: Replaces NaNs with zero and infinity with large finite numbers.
- `np.isfinite()`: Tests element-wise for finiteness (not infinity or NaN).
- `np.isnan()`: Tests element-wise for NaNs.
