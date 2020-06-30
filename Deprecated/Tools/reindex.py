import os
import argparse
from shutil import copy2

if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dirs', nargs='*', 
                        help='Specifies which directories to copy files from.')
    parser.add_argument('--output_dir', help='Specifies where to copy files to.', 
                        default=os.getcwd())
    args = parser.parse_args()

    # parser args
    input_dirs = args.input_dirs
    output_dir = args.output_dir

    # checks to see if output_dir exists, else makes the directory
    if os.path.exists(output_dir) == False:
    	os.mkdir(output_dir)

    # index for filenames
    index = 0

    # reindexes all files in input_dirs and copies them to output_dir
    for input_dir in input_dirs:
        for filename in os.listdir(input_dir):
            new_filename = "{:>09d}".format(index) + '.' + filename.split('.')[1]
            new_file = os.path.join(output_dir + r"/", new_filename)
            old_file = os.path.join(input_dir + r"/", filename)
            copy2(old_file, new_file)
            #os.rename(old_file, new_file)
            index += 1