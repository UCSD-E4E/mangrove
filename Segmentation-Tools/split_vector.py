# Given a shapefile, split into a mangrove shapefile and a nonmangrove shapefile

import fiona
import os
import argparse

def split_vector(filepath):

	dir_path = os.path.dirname(filepath)

	with fiona.open(filepath) as source:
		# Making sure m.shp does not already exist
		m_file = dir_path + "/m.shp"
		if (os.path.exists(m_file)):
			#os.remove("m.shp")	
			os.system("rm " + dir_path + "/m.*")
		# Joining mangrove polygons
		with fiona.open(m_file, "w", crs=source.crs, driver=source.driver, schema=source.schema) as m_shp:
			print("Creating mangrove files in " + dir_path)
			for record in source:
				if (record['properties']['class'] == "mangrove"):
					m_shp.write(record)
			print("Joined {} mangrove polygons.".format(len(m_shp)))

		# Making sure m.shp does not already exist
		nm_file = dir_path + "/nm.shp"
		if (os.path.exists(nm_file)):
			#os.remove("nm.shp")	
			os.system("rm " + dir_path + "/nm.*")
		# Joining mangrove polygons
		with fiona.open(nm_file, "w", crs=source.crs, driver=source.driver, schema=source.schema) as nm_shp:
			print("Creating nonmangrove files in " + dir_path)
			for record in source:
				if (record['properties']['class'] == "nonmangrove"):
					nm_shp.write(record)
			print("Joined {} nonmangrove polygons.".format(len(nm_shp)))

if __name__ == "__main__":
	# Parser to specify which .shp file to split
	parser = argparse.ArgumentParser(description='Specify .shp file to be split')
	parser.add_argument('--filepath')

	args = parser.parse_args()
	filepath = args.file_name

	split_vector(filepath)