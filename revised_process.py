import argparse
import os
import numpy as np
from PIL import Image, ImageOps
import cv2
from skimage.filters import gaussian
from wand.image import Image as WandImage
from straug.warp import Curve, Distort, Stretch
from straug.geometry import Rotate, Perspective, Shrink
from straug.pattern import Grid, VGrid, HGrid, RectGrid, EllipseGrid
from straug.weather import Fog, Snow, Frost, Rain, Shadow
from straug.camera import Contrast, Brightness, JpegCompression, Pixelate
from straug.process import Posterize, Solarize, Invert, Equalize, AutoContrast, Sharpness, Color
from straug.blur import GaussianBlur, DefocusBlur, MotionBlur, GlassBlur, ZoomBlur
import json
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger()
log_data = []

def apply_augmentation(img, func, mag):
    logger.debug(f"Applying {func.__name__} with magnitude {mag}")
    return func()(img, mag=mag)

def main():
    parser = argparse.ArgumentParser(description='Apply straug transformations to images.')
    parser.add_argument('--image_dir', required=True, help='Directory of input images')
    parser.add_argument('--result_dir', required=True, help='Directory to save augmented images')
    parser.add_argument('--width', type=int, default=200, help='Default image width')
    parser.add_argument('--height', type=int, default=64, help='Default image height')
    parser.add_argument('--seed', type=int, default=0, help='Random generator seed')
    parser.add_argument('--max_mag', type=int, default=5, help='Maximum magnitude level for augmentations')

    args = parser.parse_args()
    
    # Set the seed for reproducibility
    np.random.seed(args.seed)
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
        logger.debug(f"Created result directory: {args.result_dir}")

    # Create the log file if it does not exist
    log_file_path = os.path.join(args.result_dir, 'process_log.json')
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as log_file:
            json.dump([], log_file)
        logger.debug(f"Created log file: {log_file_path}")

    # Get all the PNG files in the input directory
    png_files = [f for f in os.listdir(args.image_dir) if f.endswith('.png')]
    logger.debug(f"Found {len(png_files)} PNG files in the input directory")

    # List of augmentation functions and their names, including blur functions with fixed multichannel issue
    augmentation_functions = [
        (Curve, "curve"), (Distort, "distort"), (Stretch, "stretch"),
        (Perspective, "perspective"), (Rotate, "rotate"), (Shrink, "shrink"),
        
        (Fog, "fog"), (Snow, "snow"), (Frost, "frost"), (Rain, "rain"), (Shadow, "shadow"),
        (Contrast, "contrast"), (Brightness, "brightness"), (JpegCompression, "jpegcompression"), (Pixelate, "pixelate"),
        (Posterize, "posterize"), (Solarize, "solarize"), (Invert, "invert"), (Equalize, "equalize"), (AutoContrast, "autocontrast"), (Sharpness, "sharpness"), (Color, "color"),
        (GaussianBlur, "gaussianblur"), (DefocusBlur, "defocusblur"), (MotionBlur, "motionblur"), (GlassBlur, "glassblur"), (ZoomBlur, "zoomblur")
    ]#(Grid, "grid"), (VGrid, "vgrid"), (HGrid, "hgrid"), (RectGrid, "rectgrid"), (EllipseGrid, "ellipsegrid"), ---add later 

    # Apply each function with magnitudes from 1 to max_mag
    for file_name in png_files:
        img_path = os.path.join(args.image_dir, file_name)
        logger.debug(f"Processing file: {img_path}")
        try:
            img = Image.open(img_path).resize((args.width, args.height))
            for func, func_name in augmentation_functions:
                for mag in range(1, args.max_mag + 1):
                    augmented_img = apply_augmentation(img, func, mag)
                    
                    # Construct the output file name
                    base_name = os.path.splitext(file_name)[0]
                    output_file_name = f"{base_name}_{func_name}_{mag}.png"
                    output_path = os.path.join(args.result_dir, output_file_name)
                    
                    # Save the augmented image
                    augmented_img.save(output_path)
                    log_entry = {
                        "file": file_name,
                        "function": func_name,
                        "magnitude": mag,
                        "output_file": output_file_name
                    }
                    log_data.append(log_entry)
        except Exception as e:
            logger.error(f"Failed to process file {file_name}: {e}")
            log_entry = {
                "file": file_name,
                "error": str(e)
            }
            log_data.append(log_entry)

    # Save log data to a JSON file
    with open(log_file_path, 'w') as log_file:
        json.dump(log_data, log_file, indent=4)
    logger.debug(f"Log data saved to {log_file_path}")

    print("Augmented images have been saved in the specified result directory.")
    print("Process log has been saved to process_log.json")

if __name__ == '__main__':
    main()
