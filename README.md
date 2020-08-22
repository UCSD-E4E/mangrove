# Mangrove Classification Using Semi Supervised Learning 

Team members:

Ashlesha Vaidya, Sidharth Suresh

Mangroves are trees and shrubs that have adapted to grow in the intertidal zone along subtropical coastlines.The main idea behind the project is to identify mangroves from the drone images. We use the semi-supervised clustering then classification approach to segment and classify images.

USAGE :

cd into the directory with code and data.

clustering.py. - python clustering.py --tilesDir \<folder with Tiles\> --outputDir \<folder to save the clustered tiles\>

Pseudo-Labelling-Classifier.py - python Pseudo-Labelling-Classifier.py


The UNET-Autoencoder is the deep learning architecture for semi-supervised segmetation of mangrove images. The Final_compiled_model.ipynb file contains all the models experimented with along with the results.
