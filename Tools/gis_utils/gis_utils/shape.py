from rasterio import windows
from rasterio.features import shapes
import multiprocess
import threading
import concurrent
import numpy as np
import geopandas as gpd
import rasterio 
import gdal 

'''

get_area

Inputs- 

img: generator from input image or orthmosaic
filename: filename of input shapefile generated from gisutils 

Outputs-
shp: geopandas dataframe of fixed polygons


'''



def get_area(gpd_df, crs='epsg:3857'):
    gpd_df.to_crs({'init': crs})
    return (m_gdf['geometry'].area).sum()


    '''
fix_gdalshp

Fixes shapefiles or geopandas databases generated from gdal_polygonize.py to output geometries that only contain the image of interest. Note that this script either takes an input geopandas database, or filename,
NOT BOTH

Inputs- 

shp: geopandas dataframe of the shapefile outputted from gdal_polygonize.py
filename: of input shapefile from gdal_polygonize.py

Outputs: 
shp: geopandas dataframe of fixed polygons

'''

def fix_gdalshp(shp=None,filename=None):
    if (type(shp) == None) and (filename != None):
        shp = geopandas.read_file(filename)
    for index, feature in shp.iterrows():
        if feature["DN"] == 0:
            shp.drop(index, inplace=True)
    if (filename != None):
        shp.to_file(filename)
    return shp



def fix_shp(shp=None,filename=None):
    if (type(shp) == None) and (filename != None):
        shp = geopandas.read_file(filename)
    for index, feature in shp.iterrows():
        if feature["raster_val"] == 0:
            shp.drop(index, inplace=True)
    
    if (filename != None):
        shp.to_file(filename)
    return shp


