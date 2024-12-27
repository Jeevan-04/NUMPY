import numpy as np
from tabulate import tabulate

# Function to print outputs in a table format
def print_table(data):
    output = data['Output']
    if isinstance(output, np.ndarray):
        output = output.tolist()
    table = [[data['Description'], data['Code'], output]]
    print(tabulate(table, headers=["Description", "Code", "Output"], tablefmt="grid"))

# 1. Print NumPy version
print_table({
    "Description": "1. NumPy Version",
    "Code": "np.__version__",
    "Output": np.__version__
})

# 2. Create and print a 0D array (scalar)
a = np.array(42)
print_table({
    "Description": "2. 0D Array (Scalar)",
    "Code": "np.array(42)",
    "Output": a
})

# 3. Create and print a 1D array (vector)
b = np.array([1, 2, 3, 4, 5])
print_table({
    "Description": "3. 1D Array (Vector)",
    "Code": "np.array([1, 2, 3, 4, 5])",
    "Output": b
})

# 4. Create and print a 2D array (matrix)
c = np.array([[1, 2, 3], [4, 5, 6]])
print_table({
    "Description": "4. 2D Array (Matrix)",
    "Code": "np.array([[1, 2, 3], [4, 5, 6]])",
    "Output": c
})

# 5. Create and print a 3D array (cube)
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
print_table({
    "Description": "5. 3D Array (Cube)",
    "Code": "np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])",
    "Output": d
})

# 6. Create and print a 5D array
arr = np.array([1, 2, 3, 4], ndmin=5)
print_table({
    "Description": "6. 5D Array",
    "Code": "np.array([1, 2, 3, 4], ndmin=5)",
    "Output": arr
})
print_table({
    "Description": "Array Type and Dimensions of 5D Array",
    "Code": "type(arr), arr.ndim",
    "Output": f"Type: {type(arr)}, Dimensions: {arr.ndim}"
})

# 7. Array Indexing
print_table({
    "Description": "7. First element of vector",
    "Code": "b[0]",
    "Output": b[0]
})
print_table({
    "Description": "Second element of vector",
    "Code": "b[1]",
    "Output": b[1]
})
print_table({
    "Description": "Sum of third and fourth elements",
    "Code": "b[2] + b[3]",
    "Output": b[2] + b[3]
})

arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print_table({
    "Description": "Element at (0,1) in matrix",
    "Code": "arr[0, 1]",
    "Output": arr[0, 1]
})
print_table({
    "Description": "Element at (1,4) in matrix",
    "Code": "arr[1, 4]",
    "Output": arr[1, 4]
})

arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print_table({
    "Description": "Element from 3D array [0,1,2]",
    "Code": "arr[0, 1, 2]",
    "Output": arr[0, 1, 2]
})
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print_table({
    "Description": "Last element from 2nd row in 2D array",
    "Code": "arr[1, -1]",
    "Output": arr[1, -1]
})

# 8. Array Slicing
print_table({
    "Description": "8. Slice from 1 to 5",
    "Code": "b[1:5]",
    "Output": b[1:5]
})
print_table({
    "Description": "Slice from 4 to end",
    "Code": "b[4:]",
    "Output": b[4:]
})
print_table({
    "Description": "Slice from start to 4",
    "Code": "b[:4]",
    "Output": b[:4]
})
print_table({
    "Description": "Slice last 3rd to last 1st",
    "Code": "b[-3:-1]",
    "Output": b[-3:-1]
})
print_table({
    "Description": "Slice from 1 to 5 with step 2",
    "Code": "b[1:5:2]",
    "Output": b[1:5:2]
})
print_table({
    "Description": "Slice every second element",
    "Code": "b[::2]",
    "Output": b[::2]
})

# 9. Array Slicing in 2D arrays
print_table({
    "Description": "9. Slice columns 1 to 4 from 2nd row",
    "Code": "arr[1, 1:4]",
    "Output": arr[1, 1:4]
})
print_table({
    "Description": "3rd column from all rows",
    "Code": "arr[:, 2]",
    "Output": arr[:, 2]
})
print_table({
    "Description": "Columns 1 to 4 from all rows",
    "Code": "arr[:, 1:4]",
    "Output": arr[:, 1:4]
})

# 10. Data Types
arr = np.array([1, 2, 3, 4])
print_table({
    "Description": "10. Data type of array with integers",
    "Code": "arr.dtype",
    "Output": arr.dtype
})

arr = np.array(['apple', 'banana', 'cherry'])
print_table({
    "Description": "Data type of array with strings",
    "Code": "arr.dtype",
    "Output": arr.dtype
})

arr = np.array([1, 2, 3, 4], dtype='S')
print_table({
    "Description": "Array with specified data type 'S'",
    "Code": "np.array([1, 2, 3, 4], dtype='S')",
    "Output": arr
})
print_table({
    "Description": "Data type of array with specified 'S'",
    "Code": "arr.dtype",
    "Output": arr.dtype
})

arr = np.array([1, 2, 3, 4], dtype='i4')
print_table({
    "Description": "Array with specified data type 'i4'",
    "Code": "np.array([1, 2, 3, 4], dtype='i4')",
    "Output": arr
})
print_table({
    "Description": "Data type of array with specified 'i4'",
    "Code": "arr.dtype",
    "Output": arr.dtype
})

# 11. Type Conversion
arr = np.array([1.1, 2.1, 3.1])
newarr = arr.astype('i')
print_table({
    "Description": "11. Convert float array to integer using 'i'",
    "Code": "arr.astype('i')",
    "Output": newarr
})
print_table({
    "Description": "Data type after conversion",
    "Code": "newarr.dtype",
    "Output": newarr.dtype
})

newarr = arr.astype(int)
print_table({
    "Description": "Convert float array to integer using int",
    "Code": "arr.astype(int)",
    "Output": newarr
})
print_table({
    "Description": "Data type after conversion",
    "Code": "newarr.dtype",
    "Output": newarr.dtype
})

arr = np.array([1, 0, 3])
newarr = arr.astype(bool)
print_table({
    "Description": "Convert integer array to boolean",
    "Code": "arr.astype(bool)",
    "Output": newarr
})
print_table({
    "Description": "Data type after conversion",
    "Code": "newarr.dtype",
    "Output": newarr.dtype
})

# 12. Copy vs View
arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
arr[0] = 42
print_table({
    "Description": "12. Copy of array",
    "Code": "arr.copy()",
    "Output": x
})
print_table({
    "Description": "Original array after modification",
    "Code": "arr[0] = 42",
    "Output": arr
})

arr = np.array([1, 2, 3, 4, 5])
x = arr.view()
arr[0] = 42
print_table({
    "Description": "View of array",
    "Code": "arr.view()",
    "Output": x
})
print_table({
    "Description": "Original array after modification",
    "Code": "arr[0] = 42",
    "Output": arr
})

x[0] = 31
print_table({
    "Description": "View after modification",
    "Code": "x[0] = 31",
    "Output": x
})
print_table({
    "Description": "Original array after view modification",
    "Code": "x[0] = 31",
    "Output": arr
})

x = arr.copy()
y = arr.view()
print_table({
    "Description": "Base of copy",
    "Code": "x.base",
    "Output": x.base
})
print_table({
    "Description": "Base of view",
    "Code": "y.base",
    "Output": y.base
})

# 13. Array Shape
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print_table({
    "Description": "13. Shape of 2D array",
    "Code": "arr.shape",
    "Output": arr.shape
})

arr = np.array([1, 2, 3, 4], ndmin=5)
print_table({
    "Description": "Shape of 5D array",
    "Code": "arr.shape",
    "Output": arr.shape
})

# 14. Reshape Array
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
newarr = arr.reshape(2, 3, 2)
print_table({
    "Description": "14. Reshape array to 2x3x2",
    "Code": "arr.reshape(2, 3, 2)",
    "Output": newarr
})

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
print_table({
    "Description": "Base of reshaped array",
    "Code": "arr.reshape(2, 4).base",
    "Output": arr.reshape(2, 4).base
})

newarr = arr.reshape(2, 2, -1)
print_table({
    "Description": "Reshape array with -1",
    "Code": "arr.reshape(2, 2, -1)",
    "Output": newarr
})

arr = np.array([[1, 2, 3], [4, 5, 6]])
newarr = arr.reshape(-1)
print_table({
    "Description": "Flatten array",
    "Code": "arr.reshape(-1)",
    "Output": newarr
})

# 15. Iterating Arrays
arr = np.array([1, 2, 3])
output = [x for x in arr]
print_table({
    "Description": "15. Iterating 1D array",
    "Code": "[x for x in arr]",
    "Output": output
})

arr = np.array([[1, 2, 3], [4, 5, 6]])
output = [x.tolist() for x in arr]
print_table({
    "Description": "Iterating 2D array",
    "Code": "[x.tolist() for x in arr]",
    "Output": output
})

output = [y for x in arr for y in x]
print_table({
    "Description": "Iterating each element in 2D array",
    "Code": "[y for x in arr for y in x]",
    "Output": output
})

arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
output = [x.tolist() for x in arr]
print_table({
    "Description": "Iterating 3D array",
    "Code": "[x.tolist() for x in arr]",
    "Output": output
})

output = [z for x in arr for y in x for z in y]
print_table({
    "Description": "Iterating each element in 3D array",
    "Code": "[z for x in arr for y in x for z in y]",
    "Output": output
})

arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
output = [x.tolist() for x in np.nditer(arr)]
print_table({
    "Description": "Iterating using nditer",
    "Code": "[x.tolist() for x in np.nditer(arr)]",
    "Output": output
})

arr = np.array([1, 2, 3])
output = [x for x in np.nditer(arr, flags=['buffered'], op_dtypes=['S'])]
print_table({
    "Description": "Iterating with different data type",
    "Code": "[x for x in np.nditer(arr, flags=['buffered'], op_dtypes=['S'])]",
    "Output": output
})

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
output = [x.tolist() for x in np.nditer(arr[:, ::2])]
print_table({
    "Description": "Iterating with step size",
    "Code": "[x.tolist() for x in np.nditer(arr[:, ::2])]",
    "Output": output
})

arr = np.array([1, 2, 3])
output = [(idx, x) for idx, x in np.ndenumerate(arr)]
print_table({
    "Description": "Iterating with index",
    "Code": "[(idx, x) for idx, x in np.ndenumerate(arr)]",
    "Output": output
})

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
output = [(idx, x) for idx, x in np.ndenumerate(arr)]
print_table({
    "Description": "Iterating 2D array with index",
    "Code": "[(idx, x) for idx, x in np.ndenumerate(arr)]",
    "Output": output
})

# 16. Joining Arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.concatenate((arr1, arr2))
print_table({
    "Description": "16. Concatenate 1D arrays",
    "Code": "np.concatenate((arr1, arr2))",
    "Output": arr
})

arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
arr = np.concatenate((arr1, arr2), axis=1)
print_table({
    "Description": "Concatenate 2D arrays along axis 1",
    "Code": "np.concatenate((arr1, arr2), axis=1)",
    "Output": arr
})

arr = np.stack((arr1, arr2), axis=1)
print_table({
    "Description": "Stack 2D arrays along axis 1",
    "Code": "np.stack((arr1, arr2), axis=1)",
    "Output": arr
})

arr = np.hstack((arr1, arr2))
print_table({
    "Description": "Horizontal stack of 2D arrays",
    "Code": "np.hstack((arr1, arr2))",
    "Output": arr
})

arr = np.vstack((arr1, arr2))
print_table({
    "Description": "Vertical stack of 2D arrays",
    "Code": "np.vstack((arr1, arr2))",
    "Output": arr
})

arr = np.dstack((arr1, arr2))
print_table({
    "Description": "Depth stack of 2D arrays",
    "Code": "np.dstack((arr1, arr2))",
    "Output": arr
})

# 17. Splitting Arrays
arr = np.array([1, 2, 3, 4, 5, 6])
newarr = np.array_split(arr, 3)
print_table({
    "Description": "17. Split 1D array into 3 parts",
    "Code": "np.array_split(arr, 3)",
    "Output": newarr
})

newarr = np.array_split(arr, 4)
print_table({
    "Description": "Split 1D array into 4 parts",
    "Code": "np.array_split(arr, 4)",
    "Output": newarr
})

print_table({
    "Description": "First part of split array",
    "Code": "newarr[0]",
    "Output": newarr[0]
})
print_table({
    "Description": "Second part of split array",
    "Code": "newarr[1]",
    "Output": newarr[1]
})
print_table({
    "Description": "Third part of split array",
    "Code": "newarr[2]",
    "Output": newarr[2]
})

arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
newarr = np.array_split(arr, 3)
print_table({
    "Description": "Split 2D array into 3 parts",
    "Code": "np.array_split(arr, 3)",
    "Output": newarr
})

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
newarr = np.array_split(arr, 3)
print_table({
    "Description": "Split 2D array into 3 parts along rows",
    "Code": "np.array_split(arr, 3)",
    "Output": newarr
})

newarr = np.array_split(arr, 3, axis=1)
print_table({
    "Description": "Split 2D array into 3 parts along columns",
    "Code": "np.array_split(arr, 3, axis=1)",
    "Output": newarr
})

newarr = np.hsplit(arr, 3)
print_table({
    "Description": "Horizontal split of 2D array",
    "Code": "np.hsplit(arr, 3)",
    "Output": newarr
})

# 18. Searching Arrays
arr = np.array([1, 2, 3, 4, 5, 4, 4])
x = np.where(arr == 4)
print_table({
    "Description": "18. Find positions of value 4",
    "Code": "np.where(arr == 4)",
    "Output": x
})

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
x = np.where(arr % 2 == 0)
print_table({
    "Description": "Find positions of even values",
    "Code": "np.where(arr % 2 == 0)",
    "Output": x
})

x = np.where(arr % 2 == 1)
print_table({
    "Description": "Find positions of odd values",
    "Code": "np.where(arr % 2 == 1)",
    "Output": x
})

arr = np.array([6, 7, 8, 9])
x = np.searchsorted(arr, 7)
print_table({
    "Description": "Find position to insert 7",
    "Code": "np.searchsorted(arr, 7)",
    "Output": x
})

x = np.searchsorted(arr, 7, side='right')
print_table({
    "Description": "Find position to insert 7 on right side",
    "Code": "np.searchsorted(arr, 7, side='right')",
    "Output": x
})

arr = np.array([1, 3, 5, 7])
x = np.searchsorted(arr, [2, 4, 6])
print_table({
    "Description": "Find positions to insert multiple values",
    "Code": "np.searchsorted(arr, [2, 4, 6])",
    "Output": x
})

# 19. Sorting Arrays
arr = np.array([3, 2, 0, 1])
print_table({
    "Description": "19. Sort integer array",
    "Code": "np.sort(arr)",
    "Output": np.sort(arr)
})

arr = np.array(['banana', 'cherry', 'apple'])
print_table({
    "Description": "Sort string array",
    "Code": "np.sort(arr)",
    "Output": np.sort(arr)
})

arr = np.array([True, False, True])
print_table({
    "Description": "Sort boolean array",
    "Code": "np.sort(arr)",
    "Output": np.sort(arr)
})

arr = np.array([[3, 2, 4], [5, 0, 1]])
print_table({
    "Description": "Sort 2D array",
    "Code": "np.sort(arr)",
    "Output": np.sort(arr)
})

# 20. Filtering Arrays
arr = np.array([41, 42, 43, 44])
x = [True, False, True, False]
newarr = arr[x]
print_table({
    "Description": "20. Filter array using boolean index list",
    "Code": "arr[x]",
    "Output": newarr
})

arr = np.array([41, 42, 43, 44])
filter_arr = []
for element in arr:
    if element > 42:
        filter_arr.append(True)
    else:
        filter_arr.append(False)
newarr = arr[filter_arr]
print_table({
    "Description": "Filter array with values > 42",
    "Code": "arr[filter_arr]",
    "Output": newarr
})

arr = np.array([1, 2, 3, 4, 5, 6, 7])
filter_arr = []
for element in arr:
    if element % 2 == 0:
        filter_arr.append(True)
    else:
        filter_arr.append(False)
newarr = arr[filter_arr]
print_table({
    "Description": "Filter array with even values",
    "Code": "arr[filter_arr]",
    "Output": newarr
})

arr = np.array([41, 42, 43, 44])
filter_arr = arr > 42
newarr = arr[filter_arr]
print_table({
    "Description": "Filter array with values > 42 using condition",
    "Code": "arr[arr > 42]",
    "Output": newarr
})

arr = np.array([1, 2, 3, 4, 5, 6, 7])
filter_arr = arr % 2 == 0
newarr = arr[filter_arr]
print_table({
    "Description": "Filter array with even values using condition",
    "Code": "arr[arr % 2 == 0]",
    "Output": newarr
})