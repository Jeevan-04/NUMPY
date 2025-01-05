import numpy as np

# Mathematical Operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("np.add:", np.add(a, b))
print("np.subtract:", np.subtract(a, b))
print("np.multiply:", np.multiply(a, b))
print("np.divide:", np.divide(a, b))
print("np.floor_divide:", np.floor_divide(a, b))
print("np.power:", np.power(a, b))
print("np.float_power:", np.float_power(a, b))
print("np.mod:", np.mod(a, b))
print("np.remainder:", np.remainder(a, b))
print("np.fmod:", np.fmod(a, b))
print("np.negative:", np.negative(a))
print("np.reciprocal:", np.reciprocal(a))

# Comparison Operations
print("np.greater:", np.greater(a, b))
print("np.greater_equal:", np.greater_equal(a, b))
print("np.less:", np.less(a, b))
print("np.less_equal:", np.less_equal(a, b))
print("np.equal:", np.equal(a, b))
print("np.not_equal:", np.not_equal(a, b))
print("np.maximum:", np.maximum(a, b))
print("np.minimum:", np.minimum(a, b))
print("np.fmax:", np.fmax(a, b))
print("np.fmin:", np.fmin(a, b))

# Logical Operations
c = np.array([True, False, True])
d = np.array([False, False, True])

print("np.logical_and:", np.logical_and(c, d))
print("np.logical_or:", np.logical_or(c, d))
print("np.logical_xor:", np.logical_xor(c, d))
print("np.logical_not:", np.logical_not(c))

# Trigonometric Functions
e = np.array([0, np.pi/2, np.pi])

print("np.sin:", np.sin(e))
print("np.cos:", np.cos(e))
print("np.tan:", np.tan(e))
print("np.arcsin:", np.arcsin(np.sin(e)))
print("np.arccos:", np.arccos(np.cos(e)))
print("np.arctan:", np.arctan(np.tan(e)))
print("np.arctan2:", np.arctan2(a, b))
print("np.hypot:", np.hypot(a, b))
print("np.degrees:", np.degrees(e))
print("np.radians:", np.radians(np.degrees(e)))
print("np.unwrap:", np.unwrap(e))

# Hyperbolic Functions
print("np.sinh:", np.sinh(e))
print("np.cosh:", np.cosh(e))
print("np.tanh:", np.tanh(e))
print("np.arcsinh:", np.arcsinh(np.sinh(e)))
print("np.arccosh:", np.arccosh(np.cosh(e)))
print("np.arctanh:", np.arctanh(np.tanh(e)))

# Exponential and Logarithmic Functions
print("np.exp:", np.exp(a))
print("np.exp2:", np.exp2(a))
print("np.expm1:", np.expm1(a))
print("np.log:", np.log(a))
print("np.log10:", np.log10(a))
print("np.log2:", np.log2(a))
print("np.log1p:", np.log1p(a))
print("np.logaddexp:", np.logaddexp(a, b))
print("np.logaddexp2:", np.logaddexp2(a, b))

# Rounding Functions
f = np.array([1.1, 2.5, 3.7])

print("np.ceil:", np.ceil(f))
print("np.floor:", np.floor(f))
print("np.rint:", np.rint(f))
print("np.round:", np.round(f))
print("np.fix:", np.fix(f))
print("np.trunc:", np.trunc(f))

# Statistical Functions
g = np.array([[1, 2, 3], [4, 5, 6]])

print("np.sum:", np.sum(g))
print("np.prod:", np.prod(g))
print("np.mean:", np.mean(g))
print("np.var:", np.var(g))
print("np.std:", np.std(g))
print("np.min:", np.min(g))
print("np.max:", np.max(g))
print("np.median:", np.median(g))
print("np.percentile:", np.percentile(g, 50))
print("np.quantile:", np.quantile(g, 0.5))

# Bitwise Operations
h = np.array([0b1100, 0b1010])
i = np.array([0b1010, 0b1100])

print("np.bitwise_and:", np.bitwise_and(h, i))
print("np.bitwise_or:", np.bitwise_or(h, i))
print("np.bitwise_xor:", np.bitwise_xor(h, i))
print("np.invert:", np.invert(h))
print("np.left_shift:", np.left_shift(h, 2))
print("np.right_shift:", np.right_shift(h, 2))

# Special Functions
j = np.array([-1.2, 1.2])

print("np.abs:", np.abs(j))
print("np.copysign:", np.copysign(j, -1))
print("np.sign:", np.sign(j))
print("np.spacing:", np.spacing(j))
print("np.sinc:", np.sinc(j))
print("np.i0:", np.i0(j))

# Custom Universal Functions
def custom_func(x):
    return x * 2

vectorized_func = np.vectorize(custom_func)
print("np.vectorize:", vectorized_func(a))

frompyfunc_func = np.frompyfunc(custom_func, 1, 1)
print("np.frompyfunc:", frompyfunc_func(a))
