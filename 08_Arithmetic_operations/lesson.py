import numpy as np

# Basic Arithmetic Operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("np.add:", np.add(a, b))
print("np.subtract:", np.subtract(a, b))
print("np.multiply:", np.multiply(a, b))
print("np.divide:", np.divide(a, b))
print("np.floor_divide:", np.floor_divide(a, b))
print("np.mod:", np.mod(a, b))
print("np.remainder:", np.remainder(a, b))
print("np.fmod:", np.fmod(a, b))
print("np.power:", np.power(a, b))
print("np.float_power:", np.float_power(a, b))
print("np.negative:", np.negative(a))
print("np.abs:", np.abs(a))

# Accumulation and Reduction
print("np.cumsum:", np.cumsum(a))
print("np.cumprod:", np.cumprod(a))
print("np.sum:", np.sum(a))
print("np.prod:", np.prod(a))
print("np.nansum:", np.nansum(a))
print("np.nanprod:", np.nanprod(a))

# Exponential and Logarithmic Operations
print("np.exp:", np.exp(a))
print("np.expm1:", np.expm1(a))
print("np.exp2:", np.exp2(a))
print("np.log:", np.log(a))
print("np.log10:", np.log10(a))
print("np.log2:", np.log2(a))
print("np.log1p:", np.log1p(a))
print("np.logaddexp:", np.logaddexp(a, b))
print("np.logaddexp2:", np.logaddexp2(a, b))

# Rounding Operations
print("np.round:", np.round(a))
print("np.around:", np.around(a))
print("np.rint:", np.rint(a))
print("np.fix:", np.fix(a))
print("np.floor:", np.floor(a))
print("np.ceil:", np.ceil(a))
print("np.trunc:", np.trunc(a))

# Special Arithmetic Functions
print("np.sqrt:", np.sqrt(a))
print("np.cbrt:", np.cbrt(a))
print("np.square:", np.square(a))
print("np.reciprocal:", np.reciprocal(a))
print("np.sign:", np.sign(a))
print("np.clip:", np.clip(a, 1, 2))
print("np.gradient:", np.gradient(a))
print("np.diff:", np.diff(a))

# Matrix-Specific Operations
c = np.array([[1, 2], [3, 4]])
d = np.array([[5, 6], [7, 8]])

print("np.matmul:", np.matmul(c, d))
print("np.dot:", np.dot(c, d))
print("np.vdot:", np.vdot(c, d))
print("np.inner:", np.inner(c, d))
print("np.outer:", np.outer(c, d))
print("np.tensordot:", np.tensordot(c, d))
print("np.kron:", np.kron(c, d))
print("np.cross:", np.cross(c, d))

# Element-wise Arithmetic with Broadcasting
print("np.add.outer:", np.add.outer(a, b))
print("np.subtract.outer:", np.subtract.outer(a, b))
print("np.multiply.outer:", np.multiply.outer(a, b))
print("np.divide.outer:", np.divide.outer(a, b))

# Logical Arithmetic Operations
print("np.bitwise_and:", np.bitwise_and(a, b))
print("np.bitwise_or:", np.bitwise_or(a, b))
print("np.bitwise_xor:", np.bitwise_xor(a, b))
print("np.bitwise_not:", np.bitwise_not(a))
print("np.left_shift:", np.left_shift(a, 1))
print("np.right_shift:", np.right_shift(a, 1))

# Comparative Arithmetic Operations
print("np.maximum:", np.maximum(a, b))
print("np.minimum:", np.minimum(a, b))
print("np.fmax:", np.fmax(a, b))
print("np.fmin:", np.fmin(a, b))

# Handling NaNs in Arithmetic
e = np.array([1, np.nan, 3])
print("np.nan_to_num:", np.nan_to_num(e))
print("np.isfinite:", np.isfinite(e))
print("np.isnan:", np.isnan(e))