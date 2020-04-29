# Given raster and shapefile, creates a pixel mask for mangrove v non-mangrove
# Only uses mangrove shape file, and considers everything else nonmangrove
# Can easily be extended to use multiple shp files by creating corresponding
# np arrays and then combining them

# https://rasterio.readthedocs.io/en/latest/topics/masking-by-shapefile.html
# https://rasterio.readthedocs.io/en/latest/topics/masks.html
# Example usage: python3 raster_mask.py --raster_filepath test_data/test.tif --vector_filepath test_data/test.shp

import fiona
import rasterio
import rasterio.mask
from matplotlib.image import imsave
import matplotlib.pyplot as plt
import numpy as np
import os
from split_vector import split_vector
import argparse

def raster_mask(raster_filepath, vector_filepath):
    # Options
    display = False

    dir_path = os.path.dirname(vector_filepath)

    # Need to run split_vector first
    m_file = os.path.join(dir_path, "m.shp")
    nm_file = os.path.join(dir_path, "nm.shp")

    if (os.path.exists(m_file) == False or os.path.exists(nm_file) == False):
        print("Splitting vectors...")
        split_vector(vector_filepath)
    else:
        print("Splits exist, skipping splitting...")

    # Opening shapefile and reading features into shapefile
    print("Reading shapefile...")
    with fiona.open(m_file, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    # Masking raster with shapefile geometries
    # All nonmangrove pixels will be set to 0, mangrove pixels
    # retain their value
    print("Creating masks...")
    with rasterio.open(raster_filepath) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=False)
        out_meta = src.meta
    
    # Creating binary mask for image
    out_mask = out_image.sum(axis=0)
    out_mask = out_mask > 0

    # Converting from color, x, y -> x, y, color for display
    if (display == True):
        display_image = np.moveaxis(out_image, (0,1,2), (2,0,1))
        plt.imshow(display_image)
        plt.show()

        plt.imshow(out_mask)
        plt.show()
        
    # Writing new raster to new file
    # Note that this mask is not a binary mask, contains all bands, but only in
    # regions specified by the shapefile features
    print("Saving masks...")
    out_meta.update({"driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})

    mask_file = os.path.join(dir_path, "mask.tif")
    with rasterio.open(mask_file, "w", **out_meta) as dest:
        dest.write(out_image)

    # Writing binary mask to new file
    binary_mask_file = os.path.join(dir_path, "mask_binary.png")
    imsave(binary_mask_file, out_mask)
    print("Done.")
    
if __name__ == "__main__":
    # Parser to specify which .shp file to split
    parser = argparse.ArgumentParser(description='Specify files for masking')
    parser.add_argument('--raster_filepath')
    parser.add_argument('--vector_filepath')

    args = parser.parse_args()
    raster_filepath = args.raster_filepath
    vector_filepath = args.vector_filepath
    raster_mask(raster_filepath, vector_filepath)