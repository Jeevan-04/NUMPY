import numpy as np

# Sample data
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
matrix = np.array([[1, 2], [3, 4]])

# Statistical Functions
print("Mean:", np.mean(data))  # Calculate mean
print("Median:", np.median(data))  # Calculate median
print("Average:", np.average(data))  # Calculate average
print("Standard Deviation:", np.std(data))  # Calculate standard deviation
print("Variance:", np.var(data))  # Calculate variance
print("Peak to Peak:", np.ptp(data))  # Calculate peak to peak
print("Minimum:", np.min(data))  # Find minimum
print("Maximum:", np.max(data))  # Find maximum
print("Percentile (50%):", np.percentile(data, 50))  # Calculate 50th percentile
print("Quantile (0.5):", np.quantile(data, 0.5))  # Calculate 0.5 quantile
print("Correlation Coefficient:\n", np.corrcoef(data))  # Calculate correlation coefficient
print("Covariance:\n", np.cov(data))  # Calculate covariance
print("Histogram:", np.histogram(data))  # Calculate histogram
print("Histogram Bin Edges:", np.histogram_bin_edges(data))  # Calculate histogram bin edges
print("Bin Count:", np.bincount(data))  # Count number of occurrences of each value
print("Digitize:", np.digitize(data, bins=[2, 4, 6, 8]))  # Return the indices of the bins to which each value belongs
print("Nan Mean:", np.nanmean(data))  # Calculate mean ignoring NaNs
print("Nan Median:", np.nanmedian(data))  # Calculate median ignoring NaNs
print("Nan Std:", np.nanstd(data))  # Calculate standard deviation ignoring NaNs
print("Nan Var:", np.nanvar(data))  # Calculate variance ignoring NaNs
print("Nan Min:", np.nanmin(data))  # Find minimum ignoring NaNs
print("Nan Max:", np.nanmax(data))  # Find maximum ignoring NaNs
print("Nan Sum:", np.nansum(data))  # Calculate sum ignoring NaNs
print("Nan Percentile (50%):", np.nanpercentile(data, 50))  # Calculate 50th percentile ignoring NaNs
print("Nan Quantile (0.5):", np.nanquantile(data, 0.5))  # Calculate 0.5 quantile ignoring NaNs

# Mathematical Functions
# Basic Arithmetic
print("Add:", np.add(1, 2))  # Add two numbers
print("Subtract:", np.subtract(2, 1))  # Subtract two numbers
print("Multiply:", np.multiply(2, 3))  # Multiply two numbers
print("Divide:", np.divide(6, 3))  # Divide two numbers
print("Floor Divide:", np.floor_divide(7, 3))  # Floor division
print("Modulus:", np.mod(7, 3))  # Modulus
print("Power:", np.power(2, 3))  # Power
print("Float Power:", np.float_power(2, 3))  # Float power
print("Remainder:", np.remainder(7, 3))  # Remainder

# Trigonometric Functions
print("Sine:", np.sin(np.pi / 2))  # Sine
print("Cosine:", np.cos(np.pi / 2))  # Cosine
print("Tangent:", np.tan(np.pi / 4))  # Tangent
print("Arcsine:", np.arcsin(1))  # Arcsine
print("Arccosine:", np.arccos(1))  # Arccosine
print("Arctangent:", np.arctan(1))  # Arctangent
print("Arctangent2:", np.arctan2(1, 1))  # Arctangent2
print("Hypotenuse:", np.hypot(3, 4))  # Hypotenuse
print("Degrees:", np.degrees(np.pi))  # Convert radians to degrees
print("Radians:", np.radians(180))  # Convert degrees to radians
print("Unwrap:", np.unwrap([0, np.pi, 2 * np.pi]))  # Unwrap

# Hyperbolic Functions
print("Hyperbolic Sine:", np.sinh(1))  # Hyperbolic sine
print("Hyperbolic Cosine:", np.cosh(1))  # Hyperbolic cosine
print("Hyperbolic Tangent:", np.tanh(1))  # Hyperbolic tangent
print("Inverse Hyperbolic Sine:", np.arcsinh(1))  # Inverse hyperbolic sine
print("Inverse Hyperbolic Cosine:", np.arccosh(1))  # Inverse hyperbolic cosine
print("Inverse Hyperbolic Tangent:", np.arctanh(0.5))  # Inverse hyperbolic tangent

# Exponential and Logarithmic Functions
print("Exponential:", np.exp(1))  # Exponential
print("Exponential Base 2:", np.exp2(1))  # Exponential base 2
print("Exponential Minus 1:", np.expm1(1))  # Exponential minus 1
print("Logarithm:", np.log(1))  # Natural logarithm
print("Logarithm Base 10:", np.log10(10))  # Logarithm base 10
print("Logarithm Base 2:", np.log2(2))  # Logarithm base 2
print("Logarithm 1 Plus:", np.log1p(1))  # Logarithm 1 plus
print("Log Add Exp:", np.logaddexp(1, 2))  # Log add exp
print("Log Add Exp Base 2:", np.logaddexp2(1, 2))  # Log add exp base 2

# Rounding
print("Ceil:", np.ceil(1.5))  # Ceil
print("Floor:", np.floor(1.5))  # Floor
print("Rint:", np.rint(1.5))  # Round to nearest integer
print("Round:", np.round(1.5))  # Round
print("Fix:", np.fix(1.5))  # Fix
print("Truncate:", np.trunc(1.5))  # Truncate

# Advanced Math Functions
print("Absolute:", np.abs(-1))  # Absolute
print("Copy Sign:", np.copysign(-1, 1))  # Copy sign
print("Sign:", np.sign(-1))  # Sign
print("Spacing:", np.spacing(1))  # Spacing
print("Sinc:", np.sinc(0))  # Sinc
print("I0:", np.i0(1))  # I0
print("Clip:", np.clip(data, 3, 7))  # Clip

# Specialized Operations
print("Product:", np.prod(data))  # Product
print("Sum:", np.sum(data))  # Sum
print("Cumulative Sum:", np.cumsum(data))  # Cumulative sum
print("Cumulative Product:", np.cumprod(data))  # Cumulative product
print("Difference:", np.diff(data))  # Difference
print("Ediff1d:", np.ediff1d(data))  # Ediff1d
print("Gradient:", np.gradient(data))  # Gradient
print("Trapezoidal:", np.trapz(data))  # Trapezoidal

# Matrix-Specific
print("Norm:", np.linalg.norm(matrix))  # Norm
print("Determinant:", np.linalg.det(matrix))  # Determinant
print("Sign and Log Determinant:", np.linalg.slogdet(matrix))  # Sign and log determinant
print("Matrix Rank:", np.linalg.matrix_rank(matrix))  # Matrix rank
print("Multi Dot:", np.linalg.multi_dot([matrix, matrix]))  # Multi dot
