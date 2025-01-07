import numpy as np
from PIL import Image
import os

# Load the image
img = Image.open("/Users/apple/Desktop/NUMPY/19_Projects/01_Image_Steganography /image.png")

# The secret message to encode
message = "Find a way not an excuse"

# Convert the image to a NumPy array
img_array = np.array(img)
print(img_array.shape)
print(img_array)

# Check if the image is in RGB format
if len(img_array.shape) < 3 or img_array.shape[2] != 3:
    raise ValueError("The input image must be in RGB format.")

# Convert the message to binary and add a stopping sequence
# ord(char): Converts each character to its ASCII code
# format(..., '08b'): Converts the ASCII code to an 8-bit binary string
# join([...]): Combines all binary strings into a single binary string
message_binary = ''.join([format(ord(char), '08b') for char in message])
# Append a stopping sequence to signal the end of the message during decoding
message_binary += '1111111111111110'  # 0xFFFE

# Check if the message fits in the image
h, w, _ = img_array.shape
max_capacity = h * w * 3  # Each pixel has 3 color channels (RGB)
if len(message_binary) > max_capacity:
    raise ValueError("The message is too long to encode in the image.")

# Flatten the image array to a 1D array
flat_img_array = img_array.flatten()

# Encode the message into the image
# Iterate over each bit in the binary message
for i, bit in enumerate(message_binary):
    # Clear the least significant bit (LSB) of the pixel value and set it to the message bit
    flat_img_array[i] = (flat_img_array[i] & 254) | int(bit)

# Reshape the encoded image array back to its original shape
encoded_img_array = flat_img_array.reshape(img_array.shape)

# Convert the NumPy array back to an image and save it
encoded_img = Image.fromarray(encoded_img_array.astype('uint8'))
encoded_img.save("/Users/apple/Desktop/NUMPY/19_Projects/01_Image_Steganography /encoded_image.png")

print("Message encoded successfully!")
