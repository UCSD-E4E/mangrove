# Introduction

Mangrove forests are highly productive ecosystems and play a vital role in carbon sequestration. Accurately monitoring their growth in an automated way has presented a challenge for many conservation groups, especially those with smaller budgets who may not have the financial resources to deploy satellites or expensive drones with multispectral sensors. Drones with cheaper sensors have the advantage of accessibility, but a streamlined workflow for drone imagery classification is still lacking. <br> <br>
This image classification tool is an accessible and automated workflow for conservation groups to quantify the amount of mangroves within their site by allowing them to upload cheaply acquired high resolution imagery (drone imagery) to a website and quantify the amount of mangroves in their site using a CNN.

</br>
<hr>

# Requirements


To run this Flask application locally, clone the Git repository and cd into the web-mangrove directory. To install all the necessary requirements (located in requirements.txt) use pip install: `pip install -r requirements.txt`

To run the server locally, use the command `python app.py`

<hr>
</br>

# File Organization

In progress..
<br>
<hr>
# Modules

## `azure_blob.py`

`azure_blob.py` is a Azure Blob store Directory Client that allows the programmer to easily interface and communicate with the Azure database. To call the function and set up a client ( a connection to the Azure database, create a client and specify the connection string and desired container to connect to. `client = azure_blob.DirectoryClient(CONNECTION_STRING, input_container)`
Then you are able to call the following functions.
<br>

| Function | Description |
| ------------ | ------------- |
| `create_blob_from_stream(blob_name, stream)` | Send .zip file to the Azure blob just from filestream (no downloading). (Note: When deploying to heroku, you may need to increase the timeout in the Procfile so if it needs more time to process these larger zip files)  |
| `upload_file(source, dest)`  | Upload a single file (source) to a path (dest) inside the container   |
| `download(source, dest)`  | Download a file or directory (source) to a path (dest) on the local filesystem   |
| `download_file(source, dest)`  | Download a single file (source) to a path (dest) on the local filesystem   |
| `ls_files(path, recursive=False)`  |  List files under a path, optionally recursively   |
| `ls_dirs(path, recursive=False)`  | List directories under a path, optionally recursively   |
| `rm(path, recursive=False)`  | Remove a single file, or remove a path recursively   |
| `rmdir(path)`  | Remove a directory and its contents recursively   |

<br>

## `classify_mod.py`
This classification module encompasses the important functions required for image classification. The primary function is `classify()` described below: 

| Function | Description |
| ------------ | ------------- |
| `classify()` | In this function, the model is downloaded from Microsoft Azure, the images are downloaded from the Microsoft Azure output-files container in batches of 32. `model.predict()` is run on these images. Then, the downsampled images are reupload to ouput-files container to replace the big files with these smaller ones  |
| `download_model(client_model)` | This function is used to download the model from Microsoft Azure. If a new model is desired and it has a different file structure, please update this function. This current file structure is ./saved_model.pb ./variables/* |

<br>


## `gdal_mergy.py`
`gdal_mergy.py` is a library containing free software to merge the tiles
<br>

<br>
## `gdal_polygonize.py`
`gdal_mergy.py` is a library containing free software to polygonize the orthomosaic
<br>
<br>

## `Procfile`
The Procfile is required for deploying to heroku. It currently contains a timeout of 200 seconds to allow for large zip files to be sent to the Azure storage via bitstream.
<br>
<br>

## `raster.py`
`raster.py` contains code from gis_utils to downsample the images

<br>
<br>

## `visualize.py`
`visualize.py` contains code to prepare the Plotly visualization. The most important functions are described below:

| Function | Description |
| ------------ | ------------- |
| `create_geojson(FILENAME, final_filename)` | This function takes in a tif filename (ex: non-mangrove classification) and creates the geojson file containing the coordinates of the polygons required for the visualization. This is done by taking in the data and converting it to a mask. From this mask, the coordinates of each shapes is stored. |
| `get_im(FILENAME, hue)` | This function takes in a tif file name (ex: non-mangrove classification 1.tif) and applies a green or red hue. Potential values for hue can be: `green_hue = (180-78)/360.0` or `red_hue = (180-180)/360.0` |


<br>
<br>
## `wsgi.py`
For `wsgi.py` be sure to run both the Flask web application with and the Dash visualization.

<br>
<br>
# Azure

| Author  | Email  |
|---|---|
| Nicole Meister  | nmeister@princeton.edu  |