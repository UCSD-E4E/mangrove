import argparse
import os
import subprocess
from tqdm import tqdm
from PIL import Image
parser = argparse.ArgumentParser(description="Retile an orthomosiac, call this in the same folder as the orthomosaic")
parser.add_argument("--width",help = "Width of output tiles")
parser.add_argument("--input", help = "input orthomosaic")
parser.add_argument("--targetDir", help = "output directory of tile imags with respect to the current directory")
args = parser.parse_args()

if args.width:
	out_width = args.width
if args.input:
	img_input = args.input
if args.targetDir:
	outputDir = args.targetDir
if args.shpfile:
	polygon = args.targetDir
else:
	polygon == False

cwd = os.getcwd()
if os.path.exists(os.path.join(cwd,outputDir)) == False:
	os.mkdir(os.path.join(cwd,outputDir))

if isinstance(polygon, str):
	polycall = "gdal -clip " + polygon + " " + img_input + " " + "clipped" + img_input
	img_input = "clipped" + img_input
	subprocess.call(polycall, shell=True)



call = "gdal_retile.py -ps " + out_width + " " + out_width + " " + "-targetDir " + outputDir + " " + img_input
print(call)
subprocess.call(call, shell=True)

img_dir = os.path.join(cwd, outputDir)
print("Removing undersized tiles in:")
print(img_dir)
for filename in tqdm(os.listdir(img_dir)):
	filepath = os.path.join(img_dir, filename)
	if os.path.splitext(filename)[1] == ".tif":
		with Image.open(filepath) as im:
			x, y = im.size
			totalsize = x*y
		if totalsize < (int(out_width) * (int(out_width))):
			os.remove(filepath)


