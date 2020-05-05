import argparse
import os
import subprocess
from osgeo import gdal,gdal_array
import numpy as np
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from PIL import Image  
import PIL  
    
    


parser = argparse.ArgumentParser(description="Cluster the tiles of an orthomosaic")
parser.add_argument("--tilesDir",help = "The directory to look for for tiles")
parser.add_argument("--outputDir" , help ="The directory to place the clustered tiles")

args = parser.parse_args()

if args.tilesDir:
    tilesDir = args.tilesDir
if args.outputDir:
    outputDir = args.outputDir
    
#Getting the list of tiles of the orthomosaic
tiles = []
for r, d, f in os.walk(tilesDir):
    for item in f:
        if '.tif' in item:
            tiles.append(os.path.join(r, item))


count=1
#Do the clustering
for tile in tiles:
    print("\n \n Processing the Tile number : ",count, "with the name :",tile)
    filename = 'clustered_'+tile.split('/')[1]
    
    
    #Reading the raster
    raster = gdal.Open(tile, gdal.GA_ReadOnly)
   
    x_size = raster.RasterXSize
    y_size = raster.RasterYSize
    band_count = raster.RasterCount#Not considering the alpha band as it represents the nodata regions
    
    #empty array to load a multiband image
    img_array = np.zeros((y_size,x_size, band_count),
               gdal_array.GDALTypeCodeToNumericTypeCode(raster.GetRasterBand(1).DataType))
    
    
    
    #loop over each band
    for b in range(img_array.shape[2]):
        img_array[:, :, b] = raster.GetRasterBand(b + 1).ReadAsArray()
   
    k = np.where(img_array[:,:,3] == 255)   #gives the pixel coordinates omit the transparent and take opaque
    im = img_array[k][:,0:3]
    row = k[0]
    col = k[1]
     
  
    print("Fitting the tile to clustering algorithm....")
    
    
    gmm = GaussianMixture(n_components=2,covariance_type = 'tied')
    gmm.fit(im)
    labels = 1-gmm.predict(im)
    
    final = np.empty((img_array.shape[0] , img_array.shape[1]))
    final[row,col] = labels 
    final = final.reshape(img_array.shape[0], img_array.shape[1])
    #plt.figure(figsize=(20,20))
    #plt.imshow(final, cmap="gray")

    #plt.show()
    
    print("Saving the output file....")
    out_im = Image.fromarray(final)
    out_im.save(outputDir +'/'+ filename)
    
    
    
    
    #USING GDAL TO WRITE IMAGES
    #ds = gdal.Open(tile, gdal.GA_ReadOnly)
    #band = ds.GetRasterBand(1)
    #arr = band.ReadAsArray()
    #[cols, rows] = arr.shape

    #driver = gdal.GetDriverByName('GTiff')
    #outRaster = driver.Create(outputDir +'/'+ filename, rows, cols, 1, gdal.GDT_Byte)
    #outRaster.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
    #outRaster.SetProjection(ds.GetProjection())##sets same projection as input
    
    
    #outRaster.GetRasterBand(1).WriteArray(final)
    
    #outRaster.FlushCache() ## remove from memory
    #del outRaster ## delete the data (not the actual geotiff)
   
    count+=1
    
    
    
    
   