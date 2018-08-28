import argparse
import os

def get_classes(input_list):
	labels = []
	for line in input_list:
		if not line.endswith(".jpg"):
			label = line.split(None,1)[0]
			if label not in labels:
				labels.append(label)
	print("Labels:" + "\n" + labels)
	return labels

def get_images(input_list):
	files = []
	for line in input_list:
		if line.endswith(".jpg"):
			files.append(line)
			

	return image, label

def mkdir(labels):
	current_directory = os.getcwd()
	for line in labels:
		final_directory = os.path.join(current_directory, line)
		if not os.path.exists(final_directory):
			os.makedirs(final_directory)
			print("Successfully made directory at" + final_directory)
		else:
			print("Directory with name " + line + " already exists!")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--images", help="images to moved")
	parser.add_argument("--labels", help="location of label file")
	args = parser.parse_args()
	if args.images:
		image_dir = args.images
	if args.labels:
		labels_dir = args.labels



	labels = get_classes(label_file)
	mkdir(labels)
	
