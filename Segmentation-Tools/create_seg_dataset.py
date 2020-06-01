# Example usage: python3 create_seg_dataset.py --map_files ../dataset/Site_1/map.txt ../dataset/Site_4/map.txt ../dataset/Site_5/map.txt


import argparse
import os
from shutil import copyfile

def create_seg_dataset(map_files):
    # Note this file structure must exist (or you can change the paths)
    # TODO add parser arguments for flexibility
    img_dir = "../dataset/training/images"
    ann_dir = "../dataset/training/annotations"

    delimeter = " -> "
    pair_id = 0

    for map_file in map_files:
        with open(map_file) as mf:
            while True:
                line = mf.readline()

                # breaks when EOF is reached
                if not line:
                    break
                
                # reading image and annotation paths
                img_path, ann_path = line.rstrip().split(sep=delimeter)

                # generating destination paths
                img_dest = os.path.join(img_dir, "image_{}.jpg".format(pair_id))
                ann_dest = os.path.join(ann_dir, "annotation_{}.jpg".format(pair_id))

                # copying files
                copyfile(img_path, img_dest)
                copyfile(ann_path, ann_dest)

                pair_id += 1 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset using map files")
    parser.add_argument("--map_files", nargs='+', help = "space separated map file paths")
    args = parser.parse_args()
    
    if args.map_files:
        map_files = args.map_files
    else:
        print("Need to specify map files, exiting.")
        exit()
    
    create_seg_dataset(map_files)