#Introduction
This UNet uses a convolutional network backbone chosen by the user to label 
pixels of an input image as either "mangrove" or "non-mangrove" 
(Semantic Segmentation)



The UNet is a group of four scripts and `unet.py`

For **training, testing, and generating predictions** one only needs to run 
`unet.py`  
<br /> 
   


#Requirements
In order to use `unet.py` or any of the associated helper scripts: 

* `create_seg_dataset.py` 
* `gen_seg_labels.py` 
* `raster_mask.py` 
* `split_vector.py`

The python libraries below must be installed:  

`keras`
`tensorflow_gpu>=2.2` or `tensorflow>=2.2` 
(this is dependent on your 
[machine/gpu](https://www.tensorflow.org/install/gpu?hl=csi))  

`segmentation_models`
`Fiona`
`rasterio`
`gdal`

<br /> 

**Note**: If you have never used the GDAL libaries before,
use the following commands:  

`apt-get install libgdal-dev`
`apt-get install python-gdal`  
<br />   

We recommend using Google Colab or an Anaconda Environment as this package also
requires many of the packages preincluded in those environments:

`tqdm`
`numpy`
`matplotlib`
`Pillow`
`joblib`  

`ml-mangrove/Segmentation/requirements.txt` will have a **version complete** 
list of the neccesary libaries  
<br />

#File Structure
In order to properly use the current build of `unet.py` and have all functions 
write to the correct directories, follow these rules and use these EXACT 
directory names:

* `unet.py` and 4 helper scripts MUST be in same directory, for this example 
it will be `/Segmentation`
* In the parent directory of `/Segmentation`, lets call it `/ml-mangrove`,
create a directory `/ml-mangrove/dataset` 
* In `/dataset`, create directories `/dataset/training` or `/dataset/testing`
depending on intended use of `unet.py` 
* For training, create `/training/images` for orthomosaics,  and 
`/training/vectors` for shapefiles 
* For testing create `/testing/images` for orthomosaics, and `/testing/output`
for output rasters

#Using the UNet
##`unet.py`  
`unet.py` takes input orthomosaic(s) and shapefile(s) pairs for training a 
binary mangrove classifier, and outputs a weight file to be used for testing. 
When testing this script takes input orthomosaic(s) and a weight file and
outputs pieces (tiles) of the original orthomosaic that have been masked
with the masks predicted by the UNet.

**Inputs:**  

`width` - size of tiles in pixels (used in retiling)

`input_rasters` - filepath(s) to orthomosaic `.tif`

`input_vectors` (required for training) - filepath(s) to shapefile for 
orthomosaic `.shp` (ordering should correspond with input rasters)  

`train`: include this flag if training the UNet

`test`: include this flag if testing the UNet

`weights`: filepath to weights file `.h5`, write location if training, or to 
use for testing 

`backbone`: name of backbone to use ex: `resnet34` or `vgg16`  

**Note:** shapefile `.shp` must be in the SAME directory as `.shx`, `.dbf`, 
`.prj`, `.qpj`, and `.cpg` files of the SAME name  
<br />

**Training the UNet**  
Example Usage:  
`python3 unet.py --width 256 --input_rasters 
../dataset/training/images/ortho1.tif ../dataset/training/images/ortho2.tif
--input_vectors ../dataset/training/vectors/shapefile1.shp  
../dataset/training/vectors/shapefile2.shp --train --weights 
../dataset/training/weights/new_weight.h5 --backbone vgg16`  

**Testing the UNet**  
Example Usage:  
`python3 unet.py --width 256 --input_rasters 
../dataset/testing/images/ortho1.tif ../dataset/testing/images/ortho2.tif
--test --weights ../dataset/testing/weights/weight_vgg16.h5 --backbone vgg16`  

#Helper Scripts
The following scripts are called upon either directly or indirectly by 
`unet.py`
<br />  

## `create_seg_dataset.py` 

`create_seg_dataset.py` uses the provided map files to place pairs of images
and annotations in their proper directories

**Inputs:** 

`map_files` - space seperated txt file(s) with image path data

`dir_name` - directory above `/images` and `/annotations` directories 
(training or testing)

`include_tif` - boolean that indicates `.tif` files are to be moved into 
`/images` as well  
<br />

**Example Usage:**  

`python3 create_seg_dataset.py --map_files ../dataset/Site_1/map.txt 
 ../dataset/Site_4/map.txt`  
<br />



## `gen_seg_labels.py`

`gen_seg_labels.py` creates `/images` and `/labels` directories, calls
`gdal_retile.py` on both the `raster_file` and `mask_file`, and creates map 
file that pairs images and labels (created during retiling). 

**Inputs:** 

`width` - size of tiles in pixels (used in retiling)

`input_raster` - filepath to orthomosaic (.tif)

`input_vector` - filepath to shapefile for orthomosaic (.shp)


`input_mask` - filepath to mask file if provided (otherwise 
`raster_mask.py` is called)

`out_dir` - directory above `/images` and `/labels` directories and map file

`convert` - boolean that indicates (retiled) labels/images are converted from 
`.tif` to `.jpg`

`destructive` - boolean that indicates `.tif` files are deleted after 
conversion to `.jpg`

**Note:** shapefile `.shp` must be in the SAME directory as `.shx`, `.dbf`, 
`.prj`, `.qpj`, and `.cpg` files of the SAME name
<br />

**Example Usage:**  

 `python3 gen_seg_labels.py --width 256 --input_raster test_data/test.tif 
 --input_mask test_data/masks/mask_binary.tif -c -d`
 
 Or with a .shp file: `python3 gen_seg_labels.py --width 256 --input_raster 
 test_data/test.tif --input_vector test_data/test.shp`  
<br />


## `raster_mask.py` 

`raster_mask.py` (if vectors not split) calls `split_vector.py` to 
seperate `input_vector` into `m.shp` and `nm.shp` and then creates a binary 
pixel mask for mangrove v non-mangrove (`mask_binary.tif` and `mask_binary.png`
). Can easily be extended to use multiple `.shp` files by creating 
corresponding numpy arrays and combining them

**Note:** this function can bog down machines with insufficent ram (<16gb) 

`rasterio` Documentation:  
[Masks](https://rasterio.readthedocs.io/en/latest/topics/masks.html)  
[Masking by shapefile](https://rasterio.readthedocs.io/en/latest/topics/masking-by-shapefile.html)


**Inputs:** 

`raster_filepath` - filepath to orthomosaic `.tif`

`vector_filepath` - filepath to original shapefile `.shp`   

**Note:** shapefile `.shp` must be in the SAME directory as `.shx`, `.dbf`, 
`.prj`, `.qpj`, and `.cpg` files of the SAME name
<br />

**Example Usage:** 

`python3 raster_mask.py --raster_filepath test_data/test.tif --vector_filepath 
test_data/test.shp`

<br />

## `split_vector.py` 

`split_vector.py` creates `/m` and `/nm` directories and splits a shapefile 
`.shp` into "mangrove" and "non-mangrove" shapefiles `.shp`.

**Inputs:** 

`filepath` - filepath to original shapefile `.shp` to be split. 

**Note:** shapefile `.shp` must be in the SAME directory as `.shx`, `.dbf`, 
`.prj`, `.qpj`, and `.cpg` files of the SAME name
<br />

**Example Usage:**  

`python3 split_vector.py test_data/test.shp`

<br />



| Author  | Email  |
|---|---|
| Sam Cole  | scole02@calpoly.edu  |


<!---

unet.py 

-->
