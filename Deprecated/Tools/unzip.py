# Used for unzipping all files in subdirectories
#Useful when downloading all of the labels which are included in zip files

import os, zipfile

cwd = os.getcwd()


for path, subdirs, files in os.walk(cwd):
	for name in files:
		
		file_name = os.path.join(path, name)
		if file_name.endswith('.zip'):
			zip_ref = zipfile.ZipFile(file_name) # create zipfile object
			zip_ref.extractall(path) # extract file to dir
			zip_ref.close() # close file
			os.remove(file_name) # delete zipped file
