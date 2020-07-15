from rasterio import windows
from rasterio.features import shapes
from rasterio.enums import Resampling
import multiprocessing
import threading
import concurrent
import numpy as np
import geopandas as gpd
import rasterio 
import gdal 
from itertools import product


'''
load_image

Inputs-

file(str): full path of image/orthmosaic

Outputs-

img: generator of original image/orthomosaic
meta(dict): contains meta information from image, including location, size, etc. 

'''

#for loading orthomosaic into memory 
def load_image(file):
    img = rasterio.open(file)
    meta = img.meta.copy()
    return img, meta



'''


#gets the windows and transforms of all tiles within an orthomosaic


get_tiles

Inputs-

ds: generator of image/orthomosaic, should be img returned from load_image()
width,height(int) = width and height of output tiles

Outputs-

out_window(list): 
out_transform(list): contains meta information from image, including location, size, etc. 

original get_tiles implementation from 

https://gis.stackexchange.com/questions/285499/how-to-split-multiband-image-into-image-tiles-using-rasterio

'''

#gets the windows and transforms of all tiles within an orthomosaic
def get_tiles(ds, width=256, height=256):
    out_window = []
    out_transform = []
    ncols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, ncols, width), range(0, nrows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=ncols, height=nrows)
    for col_off, row_off in  offsets:
        window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        out_window.append(window)
        out_transform.append(windows.transform(window, ds.transform))
    return out_window, out_transform

'''

retile

inputs:
img: generator from input image or orthmosaic
meta(dict): contains meta information from image, including location, size, etc. 

outputs(list): list of numpy arrays of input tiles

'''
def retile(img, meta, out_path = 'images/', files=False, width=256, height=256):

    #getting tiles and setting filenames for the outputy files
    output_filename = 'tile_{}-{}.tif'
    window, transform = get_tiles(img, width=width, height=height)

    #locking read and write since they are not thread safe 
    read_lock = threading.Lock()
    write_lock = threading.Lock()

    #creating process to be threaded
    def process(window,transform):

        #thread locking reading (not thread safe)
        with read_lock:
            tile = img.read(window=window)
            meta['transform'] = transform
            meta['width'], meta['height'] = window.width, window.height
        #checking for tiles that are only alpha band

        #thread locking writing (not thread safe)
        with write_lock:
            #if you want to write files 
            if files:
                outpath = os.path.join(out_path,output_filename.format(int(window.col_off), int(window.row_off)))
                with rio.open(outpath, 'w', **meta) as outds:
                    outds.write(img.read(window=window))
        return tile

    results = []

    #iterating through the different windows and tranforms generated with get_tiles
    for cur_w, cur_t in zip(window, transform):
        #running the process above with the maximum amount of workers available and returning the future (result returned) 
        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            future = executor.submit(process, cur_w, cur_t)
        results.append(future.result())
    return results

'''

polygonize

works similar to gdal_polygonize, but much faster :)

Inputs- 

img: generator from input image or orthmosaic
out_file: output filename of shapefile to write

Outputs: geopandas_df: geopandas dataframe containing the geometries from the input raster band

'''

def polygonize(img, out_file=None, band=4):
    raster = img.read(band)
    geometries = list((
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) 
        in enumerate(
            shapes(raster, mask=None, transform=img.transform))))
    
    geopandas_df  = gpd.GeoDataFrame.from_features(geometries)

    if out_file != None: 
            geopandas_df.to_file(out_file)

    return geopandas_df

'''

polygonize

works similar to gdal_polygonize, but much faster :)

Inputs- 

img: generator from input image or orthmosaic
out_file: output filename of georeferenced tif raster to write

Outputs-

resampled: numpy array containing the pixel information of the downsampled raster
transform: dict containing the transform (geolocation) data of the downsampled raster

'''


def downsample_raster(dataset, downscale_factor, out_file=None):
    
    # resample data to target shape
    resampled = dataset.read(
        out_shape=(
            dataset.count,
            int(dataset.height * downscale_factor),
            int(dataset.width * downscale_factor)
        ),
        resampling= rasterio.enums.Resampling.nearest
    )

    transform = dataset.transform * dataset.transform.scale(
        (dataset.width / resampled.shape[-1]),
        (dataset.height / resampled.shape[-2])
    )

    #if there is more than one band, output numpy array size will be (1,i,j,k), so we need to flatten the array
    if dataset.count > 1:
        resampled = resampled.squeeze()

    #writing file
    if out_file != None:
        with rasterio.open(output,'w',driver='GTiff',height=int(dataset.height * downscale_factor),width=int(dataset.width * downscale_factor),count=dataset.count,dtype=resampled.dtype,crs='+proj=latlong',transform=transform,) as dst:
            dst.write(resampled)

    return resampled, transform





def clip(shp_file, image_file):
    with fiona.open(shp_file, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
    with rasterio.open(image_file) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta
    out_meta.update({"driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform})
    return out_image, out_meta