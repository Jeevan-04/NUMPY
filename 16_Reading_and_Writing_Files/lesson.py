import numpy as np

# Create some data
data = np.array([[1, 2, 3], [4, 5, 6]])

# Save data to a text file
np.savetxt('/Users/apple/Desktop/NUMPY/16_Reading_and_Writing_Files/data.txt', data)
print("Data saved to 'data.txt' using np.savetxt")

# Load data from a text file
loaded_data = np.loadtxt('/Users/apple/Desktop/NUMPY/16_Reading_and_Writing_Files/data.txt')
print("Data loaded from 'data.txt' using np.loadtxt:")
print(loaded_data)

# Save data to a binary file in NumPy .npy format
np.save('/Users/apple/Desktop/NUMPY/16_Reading_and_Writing_Files/data.npy', data)
print("Data saved to 'data.npy' using np.save")

# Load data from a binary file in NumPy .npy format
loaded_data_npy = np.load('/Users/apple/Desktop/NUMPY/16_Reading_and_Writing_Files/data.npy')
print("Data loaded from 'data.npy' using np.load:")
print(loaded_data_npy)

# Save multiple arrays to a compressed .npz file
np.savez_compressed('/Users/apple/Desktop/NUMPY/16_Reading_and_Writing_Files/data_compressed.npz', array1=data, array2=data*2)
print("Data saved to 'data_compressed.npz' using np.savez_compressed")

# Load multiple arrays from a compressed .npz file
loaded_data_npz = np.load('/Users/apple/Desktop/NUMPY/16_Reading_and_Writing_Files/data_compressed.npz')
print("Data loaded from 'data_compressed.npz' using np.load:")
print("array1:")
print(loaded_data_npz['array1'])
print("array2:")
print(loaded_data_npz['array2'])

# Save data to a binary file
data.tofile('/Users/apple/Desktop/NUMPY/16_Reading_and_Writing_Files/data.bin')
print("Data saved to 'data.bin' using np.tofile")

# Load data from a binary file
loaded_data_bin = np.fromfile('/Users/apple/Desktop/NUMPY/16_Reading_and_Writing_Files/data.bin', dtype=int).reshape(2, 3)
print("Data loaded from 'data.bin' using np.fromfile:")
print(loaded_data_bin)

# Use np.genfromtxt to load data, handling missing values
data_with_missing = np.genfromtxt('/Users/apple/Desktop/NUMPY/16_Reading_and_Writing_Files/data.txt', missing_values='NaN', filling_values=0)
print("Data loaded from 'data.txt' using np.genfromtxt with missing values handled:")
print(data_with_missing)

# Use np.save and np.load instead of np.lib.npyio.save and np.lib.npyio.load
np.save('/Users/apple/Desktop/NUMPY/16_Reading_and_Writing_Files/data_lib.npy', data)
print("Data saved to 'data_lib.npy' using np.save")

loaded_data_lib = np.load('/Users/apple/Desktop/NUMPY/16_Reading_and_Writing_Files/data_lib.npy')
print("Data loaded from 'data_lib.npy' using np.load:")
print(loaded_data_lib)
