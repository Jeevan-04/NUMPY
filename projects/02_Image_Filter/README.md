# Image Filter Application

This project applies various filters to an image using Python, Pillow, and NumPy. The user can choose from several filters, including grayscale, chromatic, and pixelated effects.

## Project Setup

1. **Clone the Repository**: Clone this repository to your local machine.
   ```bash
   git clone https://github.com/Jeevan-04/Image-Filter.git
   cd image-Filter
   ```

2. **Install Dependencies**: Ensure you have Pillow and NumPy installed.
   ```bash
   pip install pillow numpy
   ```

## How to Use

1. **Run the Script**: Execute the `filter.py` script.
   ```bash
   python filter.py
   ```

2. **Choose a Filter**: When prompted, enter the filter you want to apply (`grayscale`, `chromatic`, `pixelated`).

## Filters

### Grayscale Filter
Converts the image to grayscale using the formula:
\[ Y = 0.2989 \times R + 0.5870 \times G + 0.1140 \times B \]

### Chromatic Filter
Creates a chromatic aberration effect by shifting the color channels:
- Red is shifted right by 2 pixels.
- Green is shifted up by 2 pixels.
- Blue remains unchanged.

### Pixelated Filter
Reduces the resolution to give a pixelated effect. The `pixel_size` parameter defines the size of each pixel block. Smaller pixel sizes keep more details while giving a pixelated appearance.

## Results

| Original Image | Grayscale Filter |
|----------------|------------------|
| ![Original Image](image.png) | ![Grayscale Filter](grayscale_filter.jpg) |

| Chromatic Filter | Pixelated Filter |
|------------------|------------------|
| ![Chromatic Filter](chromatic_filter.jpg) | ![Pixelated Filter](pixelated_filter.jpg) |

## Why This Approach?

- **Pillow**: Provides easy-to-use functions for image processing.
- **NumPy**: Efficiently handles numerical operations on image data.
- **Modular Filters**: Each filter is a separate function, making it easy to add or modify filters.

## Possible Improvements

- **More Filters**: Add more filters like sepia, blur, or edge detection.
- **GUI**: Create a graphical user interface for easier interaction.
- **Batch Processing**: Allow processing multiple images at once.

## FAQs

### What is the purpose of this project?
To demonstrate how to apply different image filters using Python.

### Can I add my own filters?
Yes, you can add new filter functions and update the `apply_filter` function to include them.

### What if I encounter an error?
Ensure the image path is correct and the required libraries are installed. Check the error message for more details.
