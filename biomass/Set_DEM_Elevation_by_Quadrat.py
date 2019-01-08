##quadrat_layer_name=vector
##ch_field_name=field quadrat_layer_name
# added
##dem_layer_name=raster

from qgis.utils import *
from qgis.core import *
from PyQt4.QtCore import *

# Reference QGIS data objects
quad_layer = processing.getObject(quadrat_layer_name)
# added
dem_layer = processing.getObject(dem_layer_name)
can_height_field = quad_layer.fieldNameIndex(ch_field_name)
# QgsRasterLayer

# Get list of DEMs
# dems = iface.legendInterface().layers()
# for dem in dems:
#    if dem.type() == QgsMapLayer.VectorLayer:
#         dems.append(dem)
#         print(dem.id())


# -- Loop this section --
# zonal_output_file
dem_count = 1

# Setup
print("Analyzing layer: " + dem_layer_name)

# Set output files
zonal_output_file = "/media/e4e/mangrove-e4e/QGIS Projects/test_output_{}".format(dem_count)
dem_output_file = "/media/e4e/mangrove-e4e/QGIS Projects/test_dem_output_{}".format(dem_count)

# Run Zonal Statistics
processing.runalg("qgis:zonalstatistics", dem_layer_name, 1, quadrat_layer_name, "_", True, zonal_output_file)

# Import output from Zonal Statistics
zonal_layer = iface.addVectorLayer(zonal_output_file + ".shp", "zonal_statistics", "ogr")

# Determine offset average
offset_sum = float(0)
offset_count = 0
for f in processing.features(zonal_layer):
    quad_height = f['Max_Height']
    dem_height = f['_max']
    if dem_height == None:
        continue
    offset = float(quad_height) - float(dem_height)
    offset_sum += offset
    offset_count += 1
    
# Remove zonal statistics layer
# print(zonal_layer.id())
# QgsMapLayerRegistry.instance().removeMapLayers(zonal_layer.id())
# os.remove(zonal_output_file + ".dbf")
# os.remove(zonal_output_file + ".prj")
# os.remove(zonal_output_file + ".qpj")
# os.remove(zonal_output_file + ".shp")
# os.remove(zonal_output_file + ".shx")
    
# Check for overlapping quadrats
if offset_count > 0:
    
    # Calculate average offset
    offset_avg = offset_sum / offset_count
    print("  Average offset: {}".format(offset_avg))
    
    # Adjust DEM
    print("Reading from file for zonalstatistics: ".format(dem_layer_name))
    processing.runalg("gdalogr:rastercalculator", dem_layer_name, 1, None, 1, None, 1, None, 1, None, 1, None, 1, "A+{}".format(offset_avg), None, 5, None, dem_output_file)
    print("Reading from file for rastercalculator: ".format(dem_output_file + ".tif"))
    # Import output from Zonal Statistics
    adj_dem_layer = iface.addRasterLayer(dem_output_file + ".tif",  "adjusted_dem", "gdal")
    
    # Generate final zonal statistics
    processing.runalg("qgis:zonalstatistics", dem_output_file + ".tif", 1, quadrat_layer_name, "_", True    , zonal_output_file)

    # Import output from Zonal Statistics
    # zonal_layer_2 = iface.addVectorLayer(zonal_output_file + "2" + ".shp", "zonal_statistics_2", "ogr")

# No overlapping quadrats found
else:
    print("  No overlapping quadrats found.")
    
