import rasterio
from shapely.geometry import box
import geopandas as gpd
from rasterio.mask import mask
from rasterio.plot import show
import os
from rasterio import windows as wind
from fiona.crs import from_epsg
from gis_utils import raster
import matplotlib.pyplot as plt
import fiona
import numpy as np
import pandas as pd


def write_ndvi(image_file):
    np.seterr(divide='ignore', invalid='ignore')
    p_img, p_meta = raster.load_image(image_file)

    red = p_img.read(1)
    nir = p_img.read(4)
    array = p_img.read()
    ndvi = (nir - red)/(nir + red)
    show(ndvi)
    p_meta.update({"count": p_img.count+1})
    out_img = np.concatenate((array, np.expand_dims(ndvi, axis=0)))

    with rasterio.open(image_file, 'w', **p_meta) as outds:
        outds.write(out_img)


def write_ndwi(image_file):
    np.seterr(divide='ignore', invalid='ignore')
    p_img, p_meta = raster.load_image(image_file)

    green = p_img.read(2)
    nir = p_img.read(4)
    array = p_img.read()
    ndwi = (green - nir)/(nir + green)
    show(ndwi)
    p_meta.update({"count": p_img.count+1})
    out_img = np.concatenate((array, np.expand_dims(ndwi, axis=0)))

    with rasterio.open(image_file, 'w', **p_meta) as outds:
        outds.write(out_img)



def write_vari(image_file):
    np.seterr(divide='ignore', invalid='ignore')
    p_img, p_meta = raster.load_image(image_file)

    red = p_img.read(1)
    green = p_img.read(2)
    blue = p_img.read(3)
    array = p_img.read()
    vari = (green - red)/(green + red - blue)
    show(vari)
    p_meta.update({"count": p_img.count+1})
    out_img = np.concatenate((array, np.expand_dims(vari, axis=0)))

    with rasterio.open(image_file, 'w', **p_meta) as outds:
        outds.write(out_img)