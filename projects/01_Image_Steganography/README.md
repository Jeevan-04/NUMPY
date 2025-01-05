# Image Steganography Project

## Table of Contents
1. [Overview](#overview)
2. [Encoding Process](#encoding-process)
3. [Decoding Process](#decoding-process)
4. [Example](#example)
   - [Encoding](#encoding)
   - [Decoding](#decoding)

## Overview
This project demonstrates how to encode and decode a secret message into an image using the least significant bit (LSB) technique.

## Encoding Process
1. **Load the Image**: The image is loaded using the PIL library and converted to a NumPy array.
2. **Prepare the Message**: The secret message is converted to a binary string. Each character is converted to its ASCII code, and then to an 8-bit binary string. A stopping sequence (`1111111111111110` or `0xFFFE`) is appended to the end of the binary message.
3. **Check Capacity**: Ensure the image has enough capacity to hold the binary message.
4. **Flatten the Image**: The image array is flattened to a 1D array for easy manipulation.
5. **Encode the Message**: The binary message is encoded into the LSBs of the image pixels.
6. **Save the Encoded Image**: The modified image array is reshaped back to its original shape and saved as a new image.

## Decoding Process
1. **Load the Encoded Image**: The encoded image is loaded and converted to a NumPy array.
2. **Flatten the Image**: The image array is flattened to a 1D array.
3. **Extract LSBs**: The LSBs of the flattened array are extracted to form the binary string of the message.
4. **Find Stopping Sequence**: The stopping sequence is located in the binary message to determine the end of the encoded message.
5. **Extract and Decode Message**: The binary message bits before the stopping sequence are extracted and converted back to text.

## Example
### Encoding
```python
import numpy as np
from PIL import Image

# Load the image
img = Image.open("/path/to/image.png")

# The secret message
message = "Find a way not an excuse"

# Convert the image to a NumPy array
img_array = np.array(img)

# Check if the image is in RGB format
if len(img_array.shape) < 3 or img_array.shape[2] != 3:
    raise ValueError("The input image must be in RGB format.")

# Convert the message to binary and add a stopping sequence
message_binary = ''.join([format(ord(char), '08b') for char in message])
message_binary += '1111111111111110'  # 0xFFFE

# Check if the message fits in the image
h, w, _ = img_array.shape
max_capacity = h * w * 3  # Each pixel has 3 color channels
if len(message_binary) > max_capacity:
    raise ValueError("The message is too long to encode in the image.")

# Flatten the image array
flat_img_array = img_array.flatten()

# Encode the message into the image
for i, bit in enumerate(message_binary):
    flat_img_array[i] = (flat_img_array[i] & 254) | int(bit)  # Clear LSB and set to message bit

# Reshape the encoded image array
encoded_img_array = flat_img_array.reshape(img_array.shape)

# Convert back to an image and save
encoded_img = Image.fromarray(encoded_img_array.astype('uint8'))
encoded_img.save("/path/to/encoded_image.png")

print("Message encoded successfully!")
```

### Decoding
```python
import numpy as np
from PIL import Image

# Load the encoded image
encoded_img = Image.open("/path/to/encoded_image.png")

# Convert the image to a NumPy array
encoded_img_array = np.array(encoded_img)

# Flatten the encoded image array
flat_encoded_img_array = encoded_img_array.flatten()

# Extract the least significant bits (LSBs)
extracted_bits = [str(flat_encoded_img_array[i] & 1) for i in range(len(flat_encoded_img_array))]

# Combine the bits to form the binary string
binary_message = ''.join(extracted_bits)

# Define the stopping sequence
stopping_sequence = '111111111111110'  # 0xFFFE

# Find the stopping sequence in the binary message
end_index = binary_message.find(stopping_sequence)
if end_index == -1:
    raise ValueError("Stopping sequence not found! Decoding failed.")

# Extract the message bits before the stopping sequence
message_bits = binary_message[:end_index]

# Convert the binary message back to text
decoded_message = ''.join(
    chr(int(message_bits[i:i + 8], 2)) for i in range(0, len(message_bits), 8)
)

print("Decoded Message:", decoded_message)
