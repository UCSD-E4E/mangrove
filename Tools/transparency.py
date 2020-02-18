from PIL import Image
import numpy as np
import sys
import os 

cwd = os.getcwd()

for subdir, dirs, files in os.walk(cwd):
	for file in files:
		filename = os.path.join(subdir, file)
		if filename.endswith('tif'):
			image = Image.open(filename)
			data = np.asarray(image)
			totalsize = data.shape[0] * data.shape[1]
			alpha = data[:,:,3]
			if np.count_nonzero(alpha) < totalsize*0.5:
				os.remove(filename)

