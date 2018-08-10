import os
file = open("labels.txt", "w") 
directory = input("Enter the directory of tif images:")
for files in os.listdir(directory):
    if files.endswith(".tif"):
    	file.write(os.path.join(directory, files) + "\n")
file.close()

