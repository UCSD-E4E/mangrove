import rasterio
import geopandas as gpd 
from shapely.geometry import box
from shapely.geometry import polygon

import os

def tiles2shp(filelist=None,shp=False,filename=None):
	if filelist == None:
		print ("No files inputted, check to make sure that filelist has a list of files set as its input")
	if ((shp == True) & (filename==None)):
		filename = "tileshp.shp"
	df = gpd.GeoDataFrame(columns=['location','geometry'])
	for file in filelist:
		if not file.endswith(".tif"):
			print("Non .tif raster included in filelist, check input for filelist")
		else:
			bounds =rasterio.open(os.path.join(dir+"/", file)).bounds
			df = df.append({'location':file, 'geometry': box(bounds[0], bounds[1], bounds[2], bounds[3])},ignore_index=True)
			df2 = gpd.geoseries.GeoSeries([geom for geom in df.unary_union.geoms])

	if shp:
		df2.to_file(filename)
	return df2

