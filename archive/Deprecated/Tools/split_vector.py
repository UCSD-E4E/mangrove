#Given a directory with a bunch of shapefiles with individual features, split all of them into individual shapefiles
#Used for splitting classed mangrove labels

import fiona
import os


cwd = os.getcwd()


for path, subdirs, files in os.walk(cwd):
	for name in files:
		file_name = os.path.join(path, name)
		dest = path
		if file_name.endswith('.shp'):
			with fiona.open(file_name) as source:
				meta = source.meta
				for f in source:
					outfile = os.path.join(dest, "%s.shp" % f['id'])
					with fiona.open(outfile, 'w', **meta) as sink:
						sink.write(f)