import os
import argparse
from pathlib import Path
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as plt_colors
import cv2

def extract_and_save_tif(input_tif_path, row, col, width, height, output_tif_path):
    """
    Extracts a rectangular region from a TIF file and saves it as a new TIF file.
    """
    try:
        with rasterio.open(input_tif_path) as src:
            # Check if the window is within the bounds of the image
            if (row < 0 or col < 0 or row + height > src.height or
                    col + width > src.width):
                raise ValueError("Extraction window is out of bounds.")

            # Read the window (region of interest)
            window = Window(col, row, width, height)
            data = src.read(window=window)

            # Get metadata from the source file to use for the output
            profile = src.profile.copy()

            # Update the profile for the new dimensions
            profile.update({
                'height': height,
                'width': width,
                'transform': rasterio.windows.transform(window, src.transform)  # Important: Update the transform!
            })
            # Write the extracted data to a new TIF file
            with rasterio.open(output_tif_path, 'w', **profile) as dst:
                dst.write(data)

        print(f"Successfully extracted region and saved to {output_tif_path}")

    except rasterio.RasterioIOError as e:
        print(f"Error opening or reading the TIF file: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def oversample_and_save(input_file, output_file_without_suffix):
    """
    Oversample the image and save
    """
    # Load the PNG image
    image = cv2.imread(input_file)

    # Check if the image was loaded successfully
    if image is None:
        raise FileNotFoundError(f"Unable to load image '{input_file}'")

    scale_factor = 2
    new_size = (image.shape[1] * scale_factor, image.shape[0] * scale_factor)

    oversampled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)

    sharpen_kernel = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
    sharpened_image = cv2.filter2D(oversampled_image, -1, sharpen_kernel)

    final_image = cv2.fastNlMeansDenoisingColored(sharpened_image, None, h=15, hColor=15, templateWindowSize=7, searchWindowSize=21)

    # Apply Gaussian Blur to smooth out blockiness
    smoothed_image = cv2.GaussianBlur(final_image, (3, 3), sigmaX=0.8)
    # Apply bilateral filter to preserve edges while reducing noise
    smoothed_image = cv2.bilateralFilter(smoothed_image, d=9, sigmaColor=75, sigmaSpace=75)

    # Save the oversampled image to a new file
    final_file_name = f"{output_file_without_suffix}.png"
    cv2.imwrite(final_file_name, smoothed_image)

    new_size = 1024
    resized_image = cv2.resize(smoothed_image, (new_size, new_size), interpolation=cv2.INTER_AREA)

    cv2.imwrite(f"{output_file_without_suffix}_resized.png", resized_image)

    cv2.imshow("Image", resized_image)

    print("Press any key to close the image window...")
    cv2.waitKey(0) 
    cv2.destroyAllWindows()  


    print(f"Oversampled image saved as '{final_file_name}'")

def extract_and_save_png(tif_path,  row, col, offset=5, band_index=1, output_png="output.png"):
    """
    """
    with rasterio.open(tif_path) as src:
        print(src.tags())
        
        red = src.read(6)
        nir = src.read(10)

        entire_img = (nir - red)/(nir + red)
        entire_img_min = np.min(entire_img)
        entire_img_max = np.max(entire_img)

        sig = src.read(window=rasterio.windows.Window(col, row, 1, 1))
        sig = np.squeeze(sig)

        # Define window boundaries
        row_start = max(0, row - offset)
        row_end = min(src.height, row + offset)
        col_start = max(0, col - offset)
        col_end = min(src.width, col + offset)

        # Read the specific band and window
        window = rasterio.windows.Window(
            col_start, row_start, col_end - col_start, row_end - row_start
        )
        block_red = src.read(6, window=window)
        block_nir = src.read(10, window=window)
                
        block_green = src.read(2 , window=window)
        block_blue = src.read(3, window=window)

        bands = [src.read(band, window=window) for band in [6,2,3]]

        block = np.stack(bands, axis=-1)
        block_ndvi = (block_nir - block_red)/(block_nir + block_red + 1e-5)
    
        # Improved Ocean Visualization
        ocean_mask = block_blue > np.percentile(block_blue, 90)  # Mask for high reflectance (water bodies)

        # Boulder Detection: High NIR Reflectance & Low NDVI
        boulder_mask = (block_nir > np.percentile(block_nir, 85)) & (block_ndvi < 0.3)

        # NDWI Calculation for Water Detection
        block_ndwi = (block_green - block_nir) / (block_green + block_nir + 1e-5)
        ocean_mask = block_ndwi > 0.3  # Threshold for water bodies

        # Normalize for visualization if needed
        block = (block - entire_img_min) / (entire_img_max - entire_img_min + 1e-5)

        # Define custom colormap for mountain, vegetation, and ocean
        # Ocean (blue), Vegetation (green), Mountain (brown)
        cmap = plt_colors.ListedColormap(['#0000FF', '#8B4513', '#00FF00' ])  
        bounds = [-1, 0.402, 0.75, 1]
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        # True Color Composite
        axs[0].imshow(block)
        axs[0].axis('off')
        axs[0].set_title('True Color Composite')

        # NDVI Heatmap
        cmap_ndvi = plt.cm.YlGn  # Colormap for vegetation
        axs[1].imshow(block_ndvi, cmap=cmap_ndvi)
        axs[1].axis('off')
        axs[1].set_title('NDVI (Vegetation)')
        
        # Save as PNG
        plt.savefig(f"{output_png}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        combined_image = np.zeros((*block_ndvi.shape, 3))

        # Apply YlGn colormap for NDVI values > 0.3
        ylgn_cmap = plt.cm.get_cmap('YlGn')
        ndvi_mask = block_ndvi > 0.3
        ndvi_normalized = (block_ndvi - 0.3) / (1 - 0.3)  # Normalize NDVI for colormap scaling
        ndvi_normalized = np.clip(ndvi_normalized, 0, 1)

        # Apply colormap to vegetation
        colored_ndvi = ylgn_cmap(ndvi_normalized)
        combined_image[ndvi_mask] = colored_ndvi[ndvi_mask, :3]

        # Apply Greys colormap for boulders
        greys_cmap = plt.cm.get_cmap('Greys')
        boulder_normalized = (block_nir - entire_img_min) / (entire_img_max - entire_img_min + 1e-5)
        colored_boulders = greys_cmap(boulder_normalized)
        combined_image[boulder_mask] = colored_boulders[boulder_mask, :3]

        # Blue tint for water bodies
        combined_image[ocean_mask, :] = [0, 0.4, 1]  

        # Ensure values are within [0, 1]
        combined_image = np.clip(combined_image, 0, 1)  

        # Save as PNG
        plt.figure(figsize=(5, 5))
        plt.imshow(combined_image)
        plt.axis('off')

        new_file_dest = f"{output_png}_combined.png"
        plt.savefig(new_file_dest, bbox_inches='tight', pad_inches=0)

        oversample_and_save(new_file_dest, f"{output_png}_image")
        
        return sig, combined_image, new_file_dest

def main(args):
    if not args.tif_file:
        raise ValueError("Please pass a tif file")
    
    tif_file = Path(args.tif_file)
    if not tif_file.exists():
        raise ValueError(f"Tif file {tif_file} does not exist.")

    row_start = 400  # Example row
    col_start = 400  # Example column
    box_width = 500
    box_height = 500

    output_file = "small_block.tif"

    extract_and_save_tif(str(tif_file), row_start, col_start, box_width, box_height, output_file)

    output_file = "small_block_enhanced.tif"
    sig, _, _ = extract_and_save_png(
        str(tif_file), row=row_start, col=col_start, offset=box_width, band_index=1, output_png=output_file
    )

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="extraction")
    parser.add_argument("tif_file", help="path to tif file")

    args = parser.parse_args()
    main(args)
