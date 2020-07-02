# Introduction

GIS Utils is a python package aimed to improve on many of the inabilities of GDAL Python scripts to be faster, properly multithreaded, and much easier to use with a ML workflow than GDAL on its own. GIS Utils provides a few high level abstractions of GIS workflows aimed to be used for ML workflows, such as tilizing of images, raster to image conversions, and a few other tools specific to our workflow in particular. 

# Requirements

In order to use GIS Utils, the below python libraries must be installed:

`rasterio`
`fiona`
`geopandas`
`gdal`

We reccomending using Google Colab or an Anaconda Environment as this package also requires many of the packages preincluded in those environments:

`tqdm`
`numpy`
`pandas`

For your convenience, we have a series of `pip` and `apt-get` commands that you can use below in order to download all of the prerequisite packages for GIS Utils. 

```console
!apt-get update
!apt-get install libgdal-dev -y
!apt-get install python-gdal -y
!apt-get install python-numpy python-scipy -y
!pip install rasterio
!pip install fiona
!pip install geopandas
```
# General Tools

## `load_image()`

`load_image` is intended to 


# Retiling Rasters


## `retile()`

`Retile` is aimed to be a faster, easier to use version of `gdal_retile.py`, being able to be used in the form of a python function, and easily transferred to be used in an application or with argparse. It has the ability to output both a numpy array of the original raster, and the ability to save the resulting tiles. `Retile` is multithreaded, meaning it is much faster than it's older gdal sibling, which unfortunately only runs on 1 thread. 

Just to give a comparison, this implementation 




#### `get_tiles`



# Polygonizing Rasters

### `polygonize()`

