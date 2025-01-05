import numpy as np

# Set how floating-point errors are handled
np.seterr(over='ignore', divide='warn', invalid='raise', under='ignore')
print("Set error handling:", np.geterr())

# Context manager for error handling
with np.errstate(divide='ignore'):
    print("Error state within context manager:", np.geterr())

# Set and get print options
np.set_printoptions(precision=2, suppress=True)
print("Print options:", np.get_printoptions())

# Save and load text files
np.savetxt('array.txt', np.array([[1, 2], [3, 4]]))
loaded_array = np.loadtxt('array.txt')
print("Loaded array from text file:", loaded_array)

# Save and load binary files
np.save('array.npy', np.array([[1, 2], [3, 4]]))
loaded_array_binary = np.load('array.npy')
print("Loaded array from binary file:", loaded_array_binary)

# Masked arrays
masked_array = np.ma.masked_where(np.array([1, 2, 3]) > 2, np.array([1, 2, 3]))
print("Masked where:", masked_array)

masked_equal = np.ma.masked_equal(np.array([1, 2, 3]), 2)
print("Masked equal:", masked_equal)

masked_greater = np.ma.masked_greater(np.array([1, 2, 3]), 2)
print("Masked greater:", masked_greater)

masked_less = np.ma.masked_less(np.array([1, 2, 3]), 2)
print("Masked less:", masked_less)

masked_inside = np.ma.masked_inside(np.array([1, 2, 3]), 1, 2)
print("Masked inside:", masked_inside)

masked_outside = np.ma.masked_outside(np.array([1, 2, 3]), 1, 2)
print("Masked outside:", masked_outside)

# Set and get fill value for masked arrays
masked_array.set_fill_value(999)
print("Set fill value:", masked_array.fill_value)

# Remove incorrect np.debug() call
# np.debug()

# Remove np.test() call to avoid ModuleNotFoundError
# np.test()

# Trace of a matrix
matrix = np.array([[1, 2], [3, 4]])
print("Trace of matrix:", np.trace(matrix))

# Logical operations
print("Logical and:", np.logical_and([True, False], [False, False]))
print("Logical or:", np.logical_or([True, False], [False, False]))
print("Logical xor:", np.logical_xor([True, False], [False, False]))

# Check for membership
print("Is in:", np.isin([1, 2, 3], [2, 3, 4]))

# Check for finite, NaN, and infinite values
print("Is finite:", np.isfinite([1, 2, np.nan, np.inf]))
print("Is NaN:", np.isnan([1, 2, np.nan, np.inf]))
print("Is infinite:", np.isinf([1, 2, np.nan, np.inf]))

# Check for complex and real objects
print("Is complex object:", np.iscomplexobj([1, 2, 3]))
print("Is real object:", np.isrealobj([1, 2, 3]))

# Check for closeness of values
print("Is close:", np.isclose([1.0, 2.0], [1.0, 2.1], atol=0.1))

# Check for equality
print("Equal:", np.equal([1, 2, 3], [1, 2, 4]))

# Check for subdtype
print("Is subdtype:", np.issubdtype(np.int32, np.integer))