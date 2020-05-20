from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from sklearn.semi_supervised import LabelSpreading,LabelPropagation
import random
from sklearn import svm
from sklearn.kernel_approximation import Nystroem
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def read_files(tilesDir,labelledTilesDir):
    tiles = []
    tile_names=[]
    for r, d, f in os.walk(tilesDir):
        for item in f:
            if '.tif' in item:
                tiles.append(os.path.join(r, item))
                tile_names.append(item)

    X_labelled=[] 
    X_unlabelled=[]
    Y_labelled=[]
    Y_unlabelled=[]
    #Read in the images 
    for index, tile in enumerate(tiles):
    
    
        labelled_raster_path = os.path.join(labelledTilesDir,"labelled_"+tile_names[index])
        if(os.path.exists(labelled_raster_path)):
            print("Procesing labelled tile...")
            lim = Image.open(labelled_raster_path)
            label = np.array(lim)
            alpha = label[:,:,3]
            r = label[:,:,0]
            alpha[alpha==255] = 1
            labels = alpha
            Y_labelled.append(labels)
        
            rim = Image.open(tile)
            raster = np.array(rim)[:,:,:3]    #It is a 4096x4096x3 image
            #gray = rgb2gray(raster)           #Converting the rgb to grayscale
            X_labelled.append(raster)
        else:
            rim = Image.open(tile)
            raster = np.array(rim)[:,:,:3]    #It is a 4096x4096x3 image
            #gray = rgb2gray(raster)           #Converting the rgb to grayscale
            X_unlabelled.append(raster)
            labels = np.full((4096, 4096), -1)
            Y_unlabelled.append(labels)
        
    X_labelled = np.asarray(X_labelled)
    Y_labelled = np.asarray(Y_labelled)
    X_unlabelled = np.asarray(X_unlabelled)
    Y_unlabelled = np.asarray(Y_unlabelled)

    X_labelled = X_labelled.reshape(len(X_labelled)*X_labelled[0].shape[0]*X_labelled[0].shape[1],len(['R','G','B']))
    Y_labelled = Y_labelled.reshape(len(Y_labelled)*Y_labelled[0].shape[0]*Y_labelled[0].shape[1],)
    X_unlabelled = X_unlabelled.reshape(len(X_unlabelled)*X_unlabelled[0].shape[0]*X_unlabelled[0].shape[1],len(['R','G','B']))
    Y_unlabelled = Y_unlabelled.reshape(len(Y_unlabelled)*Y_unlabelled[0].shape[0]*Y_unlabelled[0].shape[1],)

    return X_labelled,Y_labelled,X_unlabelled,Y_unlabelled
    
    
    
    
def check_number_distinct_labels(y):
    mangroves=0
    non_mangroves=0
    mangroves_index=[]
    non_mangroves_index=[]
    for index,i in enumerate(y):
        if i==1:
            mangroves+=1
            mangroves_index.append(index)
        if i==0:
            non_mangroves+=1
            non_mangroves_index.append(index)
    print("Number of mangrove pixels: ",mangroves)
    print("Number of non mangrove pixels: ",non_mangroves)
    return mangroves_index,non_mangroves_index




def get_random_points(mangroves_index,non_mangroves_index,length):
    random_mangroves_index = random.sample(mangroves_index, length)
    random_non_mangroves_index = random.sample(non_mangroves_index,length)
    return random_mangroves_index,random_non_mangroves_index


def form_training_sets(X,Y,random_mangroves_index,random_non_mangroves_index):
    X_train=[]
    y_train=[]
    for i in random_mangroves_index:
        X_train.append(X[i])
        y_train.append(Y[i])
    for i in random_non_mangroves_index:
        X_train.append(X[i])
        y_train.append(Y[i])
    return X_train,y_train




def train_classifier(X,y):
    clf=RandomForestClassifier()
    clf.fit(X, y)
    return clf


def create_test_sets(X,Y):
    X_test=[]
    y_test=[]
    for i in range(len(X)-1, -1, -1):
        X_test.append(X[i])
        y_test.append(Y[i])
        if len(X_test) >=10000:
            break
    return X_test,y_test


def predict_labels(clf,X):
    return clf.predict(X)



    


if __name__ == "__main__":
    #Read the image files
    tilesDir  = "/Users/ashlesha_vaidya/Desktop/UCSD Courses/CSE 237D Research/Label_Prop_Try/Tiles" 
    labelledTilesDir = "/Users/ashlesha_vaidya/Desktop/UCSD Courses/CSE 237D Research/Label_Prop_Try/Labelled_Tiles"
    
    print("Reading the original tiles and the available labelled tiles....")
    X_labelled,Y_labelled,X_unlabelled,Y_unlabelled = read_files(tilesDir,labelledTilesDir)
    
    print("Checking number of mangrove/non-mangrove tiles in labelled data....")
    mangroves_index,non_mangroves_index = check_number_distinct_labels(Y_labelled)
    
    
    random_mangroves_index,random_non_mangroves_index = get_random_points(mangroves_index,non_mangroves_index,50000)
    
    X_train,y_train = form_training_sets(X_labelled,Y_labelled,random_mangroves_index,random_non_mangroves_index)
    
    print("Fitting the classifier....")
    clf = train_classifier(X_train,y_train)
    
    print("Predicting the labels for the unlabelled pixels....")
    y_unlabelled_pred = predict_labels(clf,X_unlabelled)
    
    
    print("Retraining the model with using the unlabelled data....")
    mangroves_index,non_mangroves_index = check_number_distinct_labels(y_unlabelled_pred)
    random_mangroves_index,random_non_mangroves_index = get_random_points(mangroves_index,non_mangroves_index,50000)
    X_train_unlabelled,y_train_unlabelled = form_training_sets(X_unlabelled,y_unlabelled_pred,random_mangroves_index,random_non_mangroves_index)
    X_train.extend(X_train_unlabelled)
    y_train.extend(y_train_unlabelled)
    clf = train_classifier(X_train,y_train)
    
    print("Testing the model....")
    X_test,y_test = create_test_sets(X_labelled,Y_labelled)
    print("Performance on train sets...: ",clf.score(X_train, y_train))
    print("Performance of test sets of 10,000 samples:",metrics.accuracy_score(y_test, predict_labels(clf,X_test)))
    
          
    