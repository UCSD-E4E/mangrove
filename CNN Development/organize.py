import argparse
import os

def get_images(input_directory):
	#get list of images in directory
	images = []
	for file in os.listdir(input_directory):
		if file.endswith(".jpg"):
			images.append(file)
	return images

def mkdir(input_list):
	#get list of labels
	labels = []
	for line in input_list:
		if not line.endswith(".jpg"):
			label = line.split(None,1)[0]
			if label not in labels:
				labels.append(label)
	print("Labels:")
	print(labels)
	#make directories
	current_directory = os.getcwd()
	for line in labels:
		final_directory = os.path.join(current_directory, line)
		if not os.path.exists(final_directory):
			os.makedirs(final_directory)
			print("Successfully made directory at" + final_directory)
	return labels
		
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--images", help="directory of images to moved")
	parser.add_argument("--results", help="result file")
	args = parser.parse_args()
	if args.images:
		image_dir = args.images
	if args.results:
		result_file = args.results
	with open(result_file, "r") as file:
		annotations = [line.rstrip() for line in file]
	cwd = os.getcwd()
	labels = mkdir(annotations)
	top_k = annotations[1::len(labels)+1]
	top_anno = []
	for line in top_k:
		current_anno = line.split(None,1)[0]
		top_anno.append(current_anno)
	image_list = get_images(image_dir)
	for label, image in zip(top_anno,image_list):
		os.rename(os.path.join(cwd, image), os.path.join(cwd, label, image))
	print("Successfully moved " + str(len(image_list)) + "files" )





	



