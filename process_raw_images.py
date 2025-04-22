import os
from PIL import Image
import io
import base64

def crop_heatmap(image_path, output_path):
    """
    Crops a heatmap image to remove axes and the color legend and saves it.

    Args:
        image_path: Path to the input image file.
        output_path: Path to save the cropped image.
    """
    try:
        img = Image.open(image_path)

        # Define the bounding box for cropping
        # These coordinates might need adjustment based on the specific image
        # Left, Top, Right, Bottom
        # Inspect your image to find the appropriate coordinates
        # You might need to experiment with these values
        left = 119   # Adjust as needed
        top = 100    # Adjust as needed
        right = img.width - 93  # Adjust as needed
        bottom = img.height - 86 # Adjust as needed

        # Crop the image
        cropped_img = img.crop((left, top, right, bottom))

        # Save the cropped image
        cropped_img.save(output_path, format="PNG")
        print(f"Processed and saved: {output_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def process_images_in_folder(input_folder, output_folder):
    """
    Processes all PNG images in the input folder, crops them, and saves
    the processed images to the output folder.

    Args:
        input_folder: Path to the folder containing the input images.
        output_folder: Path to the folder where the processed images will be saved.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            input_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, f"{filename}")
            crop_heatmap(input_image_path, output_image_path)


input_image_folder = "images"  # Replace with the actual path to your images folder
output_image_folder = "processed_images"

process_images_in_folder(input_image_folder, output_image_folder)
print("Image processing complete.")
