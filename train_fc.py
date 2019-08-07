import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import numpy as np
import os, argparse, joblib
import matplotlib.pyplot as plt
import shutil

# 95%m, 97%nm site 7 recall with sparse categorical cross-entropy and 10 epochs
# 76%m, 98%nm site 8 recall w/ 256 neurons, up to ~82%m recall w/ 0.1 dropout
# presence of water as a separate class improves m recall by 1-2% on average
# goes to 83%m recall on site 8 w/ 2 dense layer-dropout pairs
# Training on site 8 is >95% for 7 and 9, but bad for the training set even w/o water
# Training on sites 4 and 8, with water from site 4, gives consistent ~94% accuracy

# When I mix all of the data together and then train on half, we get ~98% for everthing when testing
# on the other half (with water). The variance of each metric is also an order of magnitude smaller.
# Water does not seem to affect the accuracy significantly, but I should test this rigourously.
# It doesn't make a difference if multiple scalings are present, which is a bit worrying.
# Also doesn't matter if unlabeled data is mixed in as mangrove. It was site 10, which is unique, so it 
# may have only affected the site 10 classifications.

# Dropout layers reduce accuracy by about 0.4%
# Removing the second dense layer is not statistically significant
# Overall: Training set is by far the most important variable

# 310 neurons, 2 dropouts at 0.3:
# train on site7-11 gets 88-95% accuracy on site 1, which is below the Inceptionv3 96%
# train on dataset/train gets up to 93% accuracy on output
# train with extracted Inceptionv3 gets about the same all around

def remove_water(x_test, y_test, le):
    '''
    Remove vectors labeled as water from a dataset.
    '''
    if 'water' in le.classes_:
        water_index = le.transform(['water'])[0]
        return x_test[y_test!=water_index], y_test[y_test!=water_index]
    return x_test, y_test

def create_model():
    inputs = tf.keras.Input(shape=(2048,))
    x = layers.Dense(310, activation='relu')(inputs)
    x = layers.Dropout(rate=0.3, noise_shape=(310,))(x)
    x = layers.Dense(310, activation='relu')(x)
    x = layers.Dropout(rate=0.3, noise_shape=(310,))(x)
    outputs = layers.Dense(labels.shape[1], activation='softmax')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(),
        metrics=['accuracy'])
    return model
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train directory', default='output-site8/')
    parser.add_argument('--test', help='test directory')
    parser.add_argument('-r', '--retrain', action='store_true', help='retrain the model')
    parser.add_argument('--indir', help='directory of unlabeled images')
    parser.add_argument('--outdir', help='directory to output sorted images')
    parser.add_argument('--sort', action='store_true', help='sort unlabeled images')
    parser.add_argument('-v', '--validate', action='store_true', help='test on preprocessed data')
    parser.add_argument('--xv', action='store_true', help='cross-validate')
    parser.add_argument('--analyze', action='store_true')
    args = parser.parse_args()
    print('Are the train and test sets normalized the same way?')

    train_path = os.path.abspath(args.train)
    test_path = os.path.abspath(args.test)
    features = np.load(os.path.join(train_path, 'features.npy'))
    sc_train = joblib.load(os.path.join(train_path, 'sc.joblib'))
    features = sc_train.inverse_transform(features)
    labels = np.load(os.path.join(train_path, 'labels.npy'))

    # Randomize the order of the training set so that the validation set is different each time
    data = np.hstack([features, labels.reshape(-1, 1)])
    np.random.shuffle(data)
    features = data[:,:-1]
    labels = data[:,-1:]

    # Load test set
    le_train = joblib.load(os.path.join('output/', 'le.joblib'))
    x_test = np.load(os.path.join(test_path, 'features.npy'))
    y_test = np.load(os.path.join(test_path, 'labels.npy'))
    sc_test = joblib.load(os.path.join(test_path, 'sc.joblib'))
    x_test = sc_test.inverse_transform(x_test)

    y_test = np.minimum(y_test, np.ones_like(y_test))    # label water (2) as nm (1)
    labels = np.minimum(labels, np.ones_like(labels))

    oh = OneHotEncoder()
    labels = oh.fit_transform(labels.reshape(-1, 1))
    if not args.sort:
        y_test = oh.transform(y_test.reshape(-1, 1))
    if args.retrain:
        model = create_model()
        history = model.fit(features, labels, batch_size=64, epochs=15, validation_split=0.1)
    else:
        model = tf.keras.models.load_model(os.path.join(train_path, 'fc_model.h5'))
    
    if args.validate:
        y_pred = np.argmax(model.predict(x_test), axis=1)
        cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
        if np.unique(labels).shape[0] < 3:
            print(classification_report(np.argmax(y_test, axis=1), y_pred, digits=6))
            print(cm)
        else:
            # Instead of using classification report, we compute our own statistics, due to the fact that water
            # is a subclass of nm, so water-nm misclassifications don't matter.
            n_samples = y_pred.shape[0]
            m_recall = cm[0,0]/np.sum(cm[0])
            nm_recall = np.sum(cm[1:, 1:])/np.sum(cm[1]+cm[2])     # sum of nm classified as nm and nm classified as water
            m_precision = cm[0, 0]/np.sum(cm[:,0])
            nm_precision = np.sum(cm[1:, 1:])/np.sum(cm[:,1:])
            print(cm)
            print('m recall:', m_recall)
            print('m precision:', m_precision)
            print('nm recall:', nm_recall)
            print('nm precision', nm_precision)
    elif args.sort:
        in_dir = os.path.abspath(args.indir)
        if args.outdir is None:
            out_dir = in_dir    # default to same location as in_dir
        else:
            out_dir = os.path.abspath(args.outdir)
        fnames = joblib.load(os.path.join(test_path, 'fnames.joblib'))
        y_pred = np.argmax(model.predict(x_test), axis=1)
        y_labels = le_train.inverse_transform(y_pred)
        for i in range(len(fnames)):
            src = os.path.join(in_dir, fnames[i])
            dst = os.path.join(out_dir, y_labels[i], fnames[i])
            try:
                shutil.move(src, dst)
            except FileNotFoundError:
                print('no such file '+src)
    elif args.xv:
        kfold = KFold(n_splits=10)
        cvscores = []
        for train_index, test_index in kfold.split(features):
            model = create_model()
            x_train, x_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            model.fit(x_train, y_train, batch_size=64, verbose=0, epochs=10)
            scores = model.evaluate(x_test, y_test, verbose=0)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            cvscores.append(scores[1]*100)
        print('%.3f%% +/- %.3f%%' % (np.mean(cvscores), np.std(cvscores)))
    elif args.analyze:
        pca = PCA()
        pca.fit(features)
        print(np.cumsum(pca.explained_variance_ratio_))
    if args.retrain:
        model.save(os.path.join(train_path, 'fc_model.h5'))
