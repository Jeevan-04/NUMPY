| Sr no | Description                           | Code                              | Output                                       |
|-------|---------------------------------------|-----------------------------------|----------------------------------------------|
| 1     | NumPy Version                         | np.__version__                    | 2.2.1                                        |
| 2     | 0D Array (Scalar)                     | np.array(42)                      | 42                                           |
| 3     | 1D Array (Vector)                     | np.array([1, 2, 3, 4, 5])         | [1, 2, 3, 4, 5]                              |
| 4     | 2D Array (Matrix)                     | np.array([[1, 2, 3], [4, 5, 6]])  | [[1, 2, 3], [4, 5, 6]]                       |
| 5     | 3D Array (Cube)                       | np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]) | [[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]] |
| 6     | 5D Array                              | np.array([1, 2, 3, 4], ndmin=5)   | [[[[[1, 2, 3, 4]]]]]                         |
| 7     | Array Type and Dimensions of 5D Array | type(arr), arr.ndim               | Type: <class 'numpy.ndarray'>, Dimensions: 5 |
| 8     | First element of vector               | b[0]                              | 1                                            |
| 9     | Second element of vector              | b[1]                              | 2                                            |
| 10    | Sum of third and fourth elements      | b[2] + b[3]                       | 7                                            |
| 11    | Element at (0,1) in matrix            | arr[0, 1]                         | 2                                            |
| 12    | Element at (1,4) in matrix            | arr[1, 4]                         | 10                                           |
| 13    | Element from 3D array [0,1,2]         | arr[0, 1, 2]                      | 6                                            |
| 14    | Last element from 2nd row in 2D array | arr[1, -1]                        | 10                                           |
| 15    | Slice from 1 to 5                     | b[1:5]                            | [2, 3, 4, 5]                                 |
| 16    | Slice from 4 to end                   | b[4:]                             | [5]                                          |
| 17    | Slice from start to 4                 | b[:4]                             | [1, 2, 3, 4]                                 |
| 18    | Slice last 3rd to last 1st            | b[-3:-1]                          | [3, 4]                                       |
| 19    | Slice from 1 to 5 with step 2         | b[1:5:2]                          | [2, 4]                                       |
| 20    | Slice every second element            | b[::2]                            | [1, 3, 5]                                    |
| 21    | Slice columns 1 to 4 from 2nd row     | arr[1, 1:4]                       | [7, 8, 9]                                    |
| 22    | 3rd column from all rows              | arr[:, 2]                         | [3, 8]                                       |
| 23    | Columns 1 to 4 from all rows          | arr[:, 1:4]                       | [[2, 3, 4], [7, 8, 9]]                       |
| 24    | Data type of array with integers      | arr.dtype                         | int64                                        |
| 25    | Data type of array with strings       | arr.dtype                         | <U6                                          |
| 26    | Array with specified data type 'S'    | np.array([1, 2, 3, 4], dtype='S') | [b'1', b'2', b'3', b'4']                     |
| 27    | Data type of array with specified 'S' | arr.dtype                         | |S1                                          |
| 28    | Array with specified data type 'i4'   | np.array([1, 2, 3, 4], dtype='i4')| [1, 2, 3, 4]                                 |
| 29    | Data type of array with specified 'i4'| arr.dtype                         | int32                                        |
| 30    | Convert float array to integer using 'i' | arr.astype('i')                 | [1, 2, 3]                                    |
| 31    | Data type after conversion            | newarr.dtype                      | int32                                        |
| 32    | Convert float array to integer using int | arr.astype(int)                 | [1, 2, 3]                                    |
| 33    | Data type after conversion            | newarr.dtype                      | int64                                        |
| 34    | Convert integer array to boolean      | arr.astype(bool)                  | [True, False, True]                          |
| 35    | Data type after conversion            | newarr.dtype                      | bool                                         |
| 36    | Copy of array                         | arr.copy()                        | [1, 2, 3, 4, 5]                              |
| 37    | Original array after modification     | arr[0] = 42                       | [42, 2, 3, 4, 5]                             |
| 38    | View of array                         | arr.view()                        | [42, 2, 3, 4, 5]                             |
| 39    | Original array after modification     | arr[0] = 42                       | [42, 2, 3, 4, 5]                             |
| 40    | View after modification               | x[0] = 31                         | [31, 2, 3, 4, 5]                             |
| 41    | Original array after view modification| x[0] = 31                         | [31, 2, 3, 4, 5]                             |
| 42    | Base of copy                          | x.base                            |                                              |
| 43    | Base of view                          | y.base                            | [31, 2, 3, 4, 5]                             |
| 44    | Shape of 2D array                     | arr.shape                         | (2, 4)                                       |
| 45    | Shape of 5D array                     | arr.shape                         | (1, 1, 1, 1, 4)                              |
| 46    | Reshape array to 2x3x2                | arr.reshape(2, 3, 2)              | [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]] |
| 47    | Base of reshaped array                | arr.reshape(2, 4).base            | [1, 2, 3, 4, 5, 6, 7, 8]                     |
| 48    | Reshape array with -1                 | arr.reshape(2, 2, -1)             | [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]         |
| 49    | Flatten array                         | arr.reshape(-1)                   | [1, 2, 3, 4, 5, 6]                           |
| 50    | Iterating 1D array                    | [x for x in arr]                  | [np.int64(1), np.int64(2), np.int64(3)]      |
| 51    | Iterating 2D array                    | [x.tolist() for x in arr]         | [[1, 2, 3], [4, 5, 6]]                       |
| 52    | Iterating each element in 2D array    | [y for x in arr for y in x]       | [np.int64(1), np.int64(2), np.int64(3), np.int64(4), np.int64(5), np.int64(6)] |
| 53    | Iterating 3D array                    | [x.tolist() for x in arr]         | [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]] |
| 54    | Iterating each element in 3D array    | [z for x in arr for y in x for z in y] | [np.int64(1), np.int64(2), np.int64(3), np.int64(4), np.int64(5), np.int64(6), np.int64(7), np.int64(8), np.int64(9), np.int64(10), np.int64(11), np.int64(12)] |
| 55    | Iterating using nditer                | [x.tolist() for x in np.nditer(arr)] | [1, 2, 3, 4, 5, 6, 7, 8]                     |
| 56    | Iterating with different data type    | [x for x in np.nditer(arr, flags=['buffered'], op_dtypes=['S'])] | [array(b'1', dtype='|S21'), array(b'2', dtype='|S21'), array(b'3', dtype='|S21')] |
| 57    | Iterating with step size              | [x.tolist() for x in np.nditer(arr[:, ::2])] | [1, 3, 5, 7]                                 |
| 58    | Iterating with index                  | [(idx, x) for idx, x in np.ndenumerate(arr)] | [((0,), np.int64(1)), ((1,), np.int64(2)), ((2,), np.int64(3))] |
| 59    | Iterating 2D array with index         | [(idx, x) for idx, x in np.ndenumerate(arr)] | [((0, 0), np.int64(1)), ((0, 1), np.int64(2)), ((0, 2), np.int64(3)), ((0, 3), np.int64(4)), ((1, 0), np.int64(5)), ((1, 1), np.int64(6)), ((1, 2), np.int64(7)), ((1, 3), np.int64(8))] |
| 60    | Concatenate 1D arrays                 | np.concatenate((arr1, arr2))      | [1, 2, 3, 4, 5, 6]                           |
| 61    | Concatenate 2D arrays along axis 1    | np.concatenate((arr1, arr2), axis=1) | [[1, 2, 5, 6], [3, 4, 7, 8]]                |
| 62    | Stack 2D arrays along axis 1          | np.stack((arr1, arr2), axis=1)    | [[[1, 2], [5, 6]], [[3, 4], [7, 8]]]         |
| 63    | Horizontal stack of 2D arrays         | np.hstack((arr1, arr2))           | [[1, 2, 5, 6], [3, 4, 7, 8]]                 |
| 64    | Vertical stack of 2D arrays           | np.vstack((arr1, arr2))           | [[1, 2], [3, 4], [5, 6], [7, 8]]             |
| 65    | Depth stack of 2D arrays              | np.dstack((arr1, arr2))           | [[[1, 5], [2, 6]], [[3, 7], [4, 8]]]         |
| 66    | Split 1D array into 3 parts           | np.array_split(arr, 3)            | [array([1, 2]), array([3, 4]), array([5, 6])] |
| 67    | Split 1D array into 4 parts           | np.array_split(arr, 4)            | [array([1, 2]), array([3, 4]), array([5]), array([6])] |
| 68    | First part of split array             | newarr[0]                         | [1, 2]                                       |
| 69    | Second part of split array            | newarr[1]                         | [3, 4]                                       |
| 70    | Third part of split array             | newarr[2]                         | [5]                                          |
| 71    | Split 2D array into 3 parts           | np.array_split(arr, 3)            | [array([[1, 2], [3, 4]]), array([[5, 6], [7, 8]]), array([[ 9, 10], [11, 12]])] |
| 72    | Split 2D array into 3 parts along rows| np.array_split(arr, 3)            | [array([[1, 2, 3], [4, 5, 6]]), array([[ 7,  8,  9], [10, 11, 12]]), array([[13, 14, 15], [16, 17, 18]])] |
| 73    | Split 2D array into 3 parts along columns | np.array_split(arr, 3, axis=1) | [array([[ 1], [ 4], [ 7], [10], [13], [16]]), array([[ 2], [ 5], [ 8], [11], [14], [17]]), array([[ 3], [ 6], [ 9], [12], [15], [18]])] |
| 74    | Horizontal split of 2D array          | np.hsplit(arr, 3)                 | [array([[ 1], [ 4], [ 7], [10], [13], [16]]), array([[ 2], [ 5], [ 8], [11], [14], [17]]), array([[ 3], [ 6], [ 9], [12], [15], [18]])] |
| 75    | Find positions of value 4             | np.where(arr == 4)                | (array([3, 5, 6]),)                          |
| 76    | Find positions of even values         | np.where(arr % 2 == 0)            | (array([1, 3, 5, 7]),)                       |
| 77    | Find positions of odd values          | np.where(arr % 2 == 1)            | (array([0, 2, 4, 6]),)                       |
| 78    | Find position to insert 7             | np.searchsorted(arr, 7)           | 1                                            |
| 79    | Find position to insert 7 on right side | np.searchsorted(arr, 7, side='right') | 2                                        |
| 80    | Find positions to insert multiple values | np.searchsorted(arr, [2, 4, 6]) | [1, 2, 3]                                    |
| 81    | Sort integer array                    | np.sort(arr)                      | [0, 1, 2, 3]                                 |
| 82    | Sort string array                     | np.sort(arr)                      | ['apple', 'banana', 'cherry']                |
| 83    | Sort boolean array                    | np.sort(arr)                      | [False, True, True]                          |
| 84    | Sort 2D array                         | np.sort(arr)                      | [[2, 3, 4], [0, 1, 5]]                       |
| 85    | Filter array using boolean index list | arr[x]                            | [41, 43]                                     |
| 86    | Filter array with values > 42         | arr[filter_arr]                   | [43, 44]                                     |
| 87    | Filter array with even values         | arr[filter_arr]                   | [2, 4, 6]                                    |
| 88    | Filter array with values > 42 using condition | arr[arr > 42]                | [43, 44]                                     |
| 89    | Filter array with even values using condition | arr[arr % 2 == 0]            | [2, 4, 6]                                    |
