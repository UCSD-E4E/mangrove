## CNN Tile Based Classification
![alt text](images/Site5_Visualization.png)

## Tensorflow 2.0 Notebooks

Upgrades to Tensorflow 2.0 brings much more optimization compared to Tensorflow 1.0 and many more models, Seeing up to 10x increases in classification speeds, plus the code is much easier to read and know what is going on! View the below notebooks to get an intro into the code

## Retrain (TF2 Notebook)

![retrain diagram](cnn/retraindiagram.png)

Retrain is used to retrain a CNN tile classifier model using the training data uploaded to the drive. This notebook is written with Google Colab, so it makes use of drive storage for easy downloading of data and uploading of models. 

<script src="https://gist.github.com/dillhicks/1809216c8e49db9ae25e2b658b95bdd8.js"></script>

## Autoclass (TF2 Notebook)

![retrain diagram](cnn/autoclassdiagram.png)


Autoclass is used to take an input orthomosaic and output a classified orthomosaic representing the specified mangrove classes. Like Retrain, this notebook is written with Colab, making it easy to download orthomosaics and upload the classified orthomosaics.

<script src="https://gist.github.com/dillhicks/660263b03751ea3885df24be21019a57.js"></script>