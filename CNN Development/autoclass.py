from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
import subprocess
import argparse
import time
import os

### NEEDS tensorflow-gpu in venv to work (using 1.14)### 

# restricts tf debug output to terminal (set to 0 for default behavior)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_labels(label_file):
	label = []
	proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
	for l in proto_as_ascii_lines:
		label.append(l.rstrip())
	return label

def load_graph(model_file):
	graph = tf.Graph()
	graph_def = tf.compat.v1.GraphDef()
	with open(model_file, "rb") as f:
		graph_def.ParseFromString(f.read())
	with graph.as_default():
		tf.import_graph_def(graph_def)
		# locks graph to prevent new operations from being added
		graph.finalize()
	return graph


def read_tensor_from_image_file(file_name):
	input_name = "file_reader"

	# adding data processing pipeline to CPU explicitly
	with tf.device('/cpu:0'):
		file_reader = tf.io.read_file(file_name, input_name)
		if file_name.endswith(".png"):
			image_reader = tf.image.decode_png(file_reader, channels=3, name="png_reader")
		elif file_name.endswith(".gif"):
			image_reader = tf.squeeze(tf.image.decode_gif(file_reader, name="gif_reader"))
		elif file_name.endswith(".bmp"):
			image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
		else:
			image_reader = tf.image.decode_jpeg(file_reader, channels=3, name="jpeg_reader")
			float_caster = tf.cast(image_reader, tf.float32)
	# tensor dimensions (299, 299, 3)
	return float_caster

def preprocess_image_batch(batch,
						input_height=299,
						input_width=299,
						input_mean=0,
						input_std=255):
	resized = tf.compat.v1.image.resize_bilinear(batch, [input_height, input_width])
	normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
	sess = tf.compat.v1.Session()
	result = sess.run(normalized)
	return result

def generate_batches(filenames, batch_size):
	# list of lists of size batch_size containing file names
 	batches = [filenames[i * batch_size:(i + 1) * batch_size] for i in range((len(filenames) + batch_size - 1) // batch_size)]  
 	num_batches = len(batches)
 	return batches, num_batches

if __name__ == "__main__":
	# for performance analysis
	start_time = time.time()
	#default args
	file_name = "tensorflow/examples/label_image/data/grace_hopper.jpg"
	model_file = \
	"tensorflow/examples/label_image/data/inception_v3_2016_08_28_frozen.pb"
	label_file = "tensorflow/examples/label_image/data/imagenet_slim_labels.txt"
	input_height = 299
	input_width = 299
	input_mean = 0
	input_std = 255
	input_layer = "input"
	output_layer = "InceptionV3/Predictions/Reshape_1"
	#add arguments for classifier
	parser = argparse.ArgumentParser()
	parser.add_argument("--images", help="image directory to be processed, must be jpg's")
	parser.add_argument("--graph", help="graph/model to be executed")
	parser.add_argument("--labels", help="name of file containing labels")
	parser.add_argument("--output_file", help="path and name with which to write result file")
	parser.add_argument("--batch_size", help="image batch size")
	parser.add_argument("--input_height", type=int, help="input height")
	parser.add_argument("--input_width", type=int, help="input width")
	parser.add_argument("--input_mean", type=int, help="input mean")
	parser.add_argument("--input_std", type=int, help="input std")
	parser.add_argument("--input_layer", help="name of input layer")
	parser.add_argument("--output_layer", help="name of output layer")
	args = parser.parse_args()

	if args.graph:
		model_file = args.graph
	if args.images:
		image_directory = args.images
	if args.labels:
		label_file = args.labels
	if args.output_file:
		output_file = args.output_file
	else:
		output_file = "results.csv"
	if args.batch_size:
		batch_size = int(args.batch_size)
	else:
		batch_size = 128 # default
	if args.input_height:
		input_height = args.input_height
	if args.input_width:
		input_width = args.input_width
	if args.input_mean:
		input_mean = args.input_mean
	if args.input_std:
		input_std = args.input_std
	if args.input_layer:
		input_layer = args.input_layer
	else:
		input_layer = "Placeholder"
	if args.output_layer:
		output_layer = args.output_layer
	else:
		output_layer = "final_result"

	# Loading tf graph and creating list of files to process
	graph = load_graph(model_file)
	file_list = []
	for root, dirs, files in os.walk(os.path.abspath(image_directory)):
		for file in files:
						file_list.append(os.path.join(root, file))

	input_name = "import/" + input_layer
	output_name = "import/" + output_layer
	input_operation = graph.get_operation_by_name(input_name)
	output_operation = graph.get_operation_by_name(output_name)

	# generating batches
	filenames = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if not file.endswith(".tif")]
	batches, num_batches = generate_batches(filenames, batch_size)
	
	# for tracking execution time
	count = 0
	batch_time = time.time()
	batch_times = []
	#load labels and create
	labels = load_labels(label_file)
	labels.append('file')
	result_df = pd.DataFrame(columns=labels)
	cwd = os.getcwd()

	# preprocess imagery
	for i in tqdm(range(num_batches)):
		files = batches[i]
		image_batch = list(map(read_tensor_from_image_file, batches[i]))
		image_batch = preprocess_image_batch(image_batch)

		with tf.compat.v1.Session(graph=graph) as sess:
			results = sess.run(output_operation.outputs[0], {
				input_operation.outputs[0]: image_batch})
			# writing result files
			file_num = 0
			for result in results:
				cur_result = pd.DataFrame(columns=labels)
				top_k = result.argsort()[-5:][::-1]
				cur_result.at[0, 'file'] = files[file_num]  
				file_num += 1
				for i in top_k:
					#print(labels[i], result[i])
					cur_result.at[0, labels[i]] =  result[i]
				#add and write result to the csv
				result_df = result_df.append(cur_result)
				result_df.to_csv(output_file)
			# for logs
			num_processed = len(image_batch)
			count += num_processed
			batch_times.append(time.time()-batch_time)
			batch_time = time.time()

	print("Script took %s seconds to execute" % (time.time() - start_time))
	print("Batch times: ")
	print(batch_times)
	labels.remove('file')
	print("Finding highest probability classes")
	result_df[labels] = result_df[labels].apply(pd.to_numeric, errors='coerce')
	result_df['max'] = result_df[labels].idxmax(axis=1)
	result_df.to_csv(output_file)
	print("Done!")
	
	dest_folders = []
	#Organize tiles into folders
	for index, row in tqdm(result_df.iterrows()):
		cur_file = row['file']
		cur_file = cur_file.replace("jpg","tif",2)
		classification = row['max'] 
		dest_folder = os.path.join(os.path.abspath(image_directory),classification)
		dest_folders.append(dest_folder)
		if os.path.exists(dest_folder) == False:
			os.mkdir(dest_folder)
		dest = os.path.join(dest_folder,os.path.basename(cur_file))
		src = cur_file
		os.rename(src, dest)
	
	for dest in dest_folders:
		call = "gdal_merge.py -o " + image_directory + " " + os.path.join(dest, "*")
		print(call)
		subprocess.call(call, shell=True)
	

	


	



    
