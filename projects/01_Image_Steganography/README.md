# Image Steganography Project

## Table of Contents
1. [Overview](#overview)
2. [Encoding Process](#encoding-process)
3. [Decoding Process](#decoding-process)
4. [Results](#results)

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

## Results
| Original Image | Encoded Image |
|----------------|---------------|
| ![Original Image](image.png) | ![Encoded Image](encoded_image.png) |
|Encoded Message: Find a way not an excuse |Decoded message: Find a way not an excuse |

