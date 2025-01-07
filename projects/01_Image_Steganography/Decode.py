import numpy as np
from PIL import Image

# Load the encoded image
encoded_img = Image.open("/Users/apple/Desktop/NUMPY/19_Projects/01_Image_Steganography /encoded_image.png")

# Convert the image to a NumPy array
encoded_img_array = np.array(encoded_img)

# Flatten the encoded image array to a 1D array
flat_encoded_img_array = encoded_img_array.flatten()

# Extract the least significant bits (LSBs) from the flattened array
# The LSBs contain the encoded message bits
extracted_bits = [str(flat_encoded_img_array[i] & 1) for i in range(len(flat_encoded_img_array))]

# Combine the extracted bits to form the binary string of the message
binary_message = ''.join(extracted_bits)

# Define the stopping sequence used during encoding
stopping_sequence = '111111111111110'  # 0xFFFE

# Find the stopping sequence in the binary message
end_index = binary_message.find(stopping_sequence)
if end_index == -1:
    raise ValueError("Stopping sequence not found! Decoding failed.")

# Extract the message bits before the stopping sequence
message_bits = binary_message[:end_index]

# Convert the binary message back to text
# Iterate over each 8-bit chunk and convert it to the corresponding character
decoded_message = ''.join(
    chr(int(message_bits[i:i + 8], 2)) for i in range(0, len(message_bits), 8)
)

print("Decoded Message:", decoded_message)
