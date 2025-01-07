from PIL import Image  # Import the Pillow library for image processing
import numpy as np  # Import NumPy for numerical operations on image data

# Load the image
image_path = "/Users/apple/Desktop/NUMPY/19_Projects/02_Image_Filter/image.png"  # Path to your input image
img = Image.open(image_path)  # Open the image
img = img.convert("RGB")  # Convert the image to RGB mode to ensure compatibility

# Convert image to NumPy array
img_array = np.array(img)  # Convert the image to a NumPy array for easy manipulation

# Grayscale Filter: Convert the image to grayscale
def grayscale_filter(img_array):
    """
    Converts the RGB image to grayscale using a weighted sum formula:
    Y = 0.2989 * R + 0.5870 * G + 0.1140 * B
    """
    return np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

# Chromatic Filter: Add an RGB split effect for a vibrant look
def chromatic_filter(img_array):
    """
    Creates a chromatic aberration effect by shifting color channels:
    - Red is shifted right by 2 pixels
    - Green is shifted up by 2 pixels
    - Blue remains unchanged
    """
    red_channel = np.roll(img_array[:, :, 0], shift=2, axis=1)  # Shift red channel right
    green_channel = np.roll(img_array[:, :, 1], shift=-2, axis=0)  # Shift green channel up
    blue_channel = img_array[:, :, 2]  # Keep the blue channel unchanged

    # Stack the channels back together
    return np.stack((red_channel, green_channel, blue_channel), axis=2)

# Pixelated Filter: Reduces resolution to give a pixelated effect with finer details
def pixelated_filter(img_array, pixel_size=5):
    """
    Creates a pixelated effect by resizing the image to a smaller size and back:
    - `pixel_size` defines the size of each pixel block. 
    - Smaller pixel size keeps more details while giving a pixelated appearance.
    """
    height, width, _ = img_array.shape
    # Resize to smaller dimensions (higher resolution with smaller pixels)
    small_img = Image.fromarray(img_array).resize((width // pixel_size, height // pixel_size), Image.NEAREST)
    # Resize back to original dimensions
    pixelated_img = small_img.resize((width, height), Image.NEAREST)
    return np.array(pixelated_img)

# Apply filters based on user input
def apply_filter(filter_name, img_array):
    """
    Applies the selected filter based on the user's choice.
    """
    if filter_name == "grayscale":
        return grayscale_filter(img_array)
    elif filter_name == "chromatic":
        return chromatic_filter(img_array)
    elif filter_name == "pixelated":
        return pixelated_filter(img_array)
    else:
        raise ValueError("Unknown filter name. Available filters: grayscale, chromatic, pixelated")

# Get user input
filter_name = input("Enter the filter you want to apply (grayscale, chromatic, pixelated): ").strip().lower()

# Apply the selected filter
try:
    # Apply the chosen filter and save the output
    filtered_img_array = apply_filter(filter_name, img_array.copy())
    filtered_img = Image.fromarray(filtered_img_array)
    output_path = f"{filter_name}_filter.jpg"
    filtered_img.save("/Users/apple/Desktop/NUMPY/19_Projects/02_Image_Filter/"+ output_path)
    print(f"{filter_name.capitalize()} filter applied and saved as '{output_path}'")
except ValueError as e:
    print(e)
