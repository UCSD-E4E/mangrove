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
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import time

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
    mangroves_index = np.where(y == 1)
    non_mangroves_index = np.where(y == 0)
    mangroves = len(mangroves_index[0])
    non_mangroves = len(non_mangroves_index[0])
    print("Number of mangrove pixels: ",mangroves)
    print("Number of non mangrove pixels: ",non_mangroves)
    return list(mangroves_index[0]),list(non_mangroves_index[0])




def get_random_points(mangroves_index,non_mangroves_index,length):
    random_mangroves_index = random.sample(mangroves_index, length)
    random_non_mangroves_index = random.sample(non_mangroves_index,length)
    return random_mangroves_index,random_non_mangroves_index


def form_training_sets(X,Y,random_mangroves_index,random_non_mangroves_index):
    X_t1 = list(X[random_mangroves_index])
    y_t1 = list(Y[random_mangroves_index])
    X_t2 =list(X[random_non_mangroves_index])
    y_t2 = list(Y[random_non_mangroves_index])
    return X_t1 + X_t2 , y_t1 + y_t2



def train_classifier(X,y,clf):
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
    
    
    mangroves_index_labelled,non_mangroves_index_labelled = check_number_distinct_labels(Y_labelled)
    random_mangroves_index_labelled,random_non_mangroves_index_labelled = get_random_points(mangroves_index_labelled,non_mangroves_index_labelled,50000)
    
    classifiers = [
    #KNeighborsClassifier(3),
    #SVC(kernel="linear", C=0.025),
    #SVC(gamma=2, C=1),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    #DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10),
    #MLPClassifier(alpha=1, max_iter=1000),
    #AdaBoostClassifier(),
    #GaussianNB(),
    #QuadraticDiscriminantAnalysis()
    ]
    
    for classifier in classifiers: 
        X_train,y_train = form_training_sets(X_labelled,Y_labelled,random_mangroves_index_labelled,random_non_mangroves_index_labelled)
        start = time.time()
        print("#"*50)
        print("#"*50)
        
        print("Fitting the ", classifier," classifier..........")
        clf = train_classifier(X_train,y_train,classifier)
        print("Predicting the labels for the unlabelled pixels....")
        y_unlabelled_pred = predict_labels(clf,X_unlabelled)
    
    
        print("Retraining the model with using the unlabelled data....")
        mangroves_index,non_mangroves_index = check_number_distinct_labels(y_unlabelled_pred)
        random_mangroves_index,random_non_mangroves_index = get_random_points(mangroves_index,non_mangroves_index,50000)
        X_train_unlabelled,y_train_unlabelled = form_training_sets(X_unlabelled,y_unlabelled_pred,random_mangroves_index,random_non_mangroves_index)
        X_train.extend(X_train_unlabelled)
        y_train.extend(y_train_unlabelled)
        clf = train_classifier(X_train,y_train,classifier)
    
        print("Testing the model.................")
        X_test,y_test = create_test_sets(X_labelled,Y_labelled)
        print("Performance on train sets...: ",clf.score(X_train, y_train))
        print("Performance of test sets of 10,000 samples:",metrics.accuracy_score(y_test, predict_labels(clf,X_test)))
        end = time.time()
        print("The time elapsed for ",classifier,"classifier .........:",end - start)
    
          
    
