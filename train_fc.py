import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import numpy as np
import os, argparse, joblib
import shutil
from tqdm import tqdm
import yaml
import sys

def create_model(input_size):
    '''
    Create an MLP model with input of a given size.

    Arguments:
        input_size: the length of the input vector
    
    Returns:
        Model: An MLP model
    '''
    inputs = tf.keras.Input(shape=(input_size,))
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dropout(rate=0.5, noise_shape=(256,))(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(momentum=0.1),
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
    parser.add_argument('--cfg', help='config file')
    args = parser.parse_args()
    cfg = yaml.load(open(args.cfg, 'r'), Loader=yaml.SafeLoader)
    train_paths = cfg['train']
    train_path = train_paths[0]
    test_path = os.path.abspath(args.test)

    # Load test set
    le_train = joblib.load(os.path.join(train_path, 'le.joblib'))
    x_test = np.load(os.path.join(test_path, 'features.npy'))
    y_test = np.load(os.path.join(test_path, 'labels.npy'))

    # Load the training set. The data from each folder will be un-normalized with the scaler if it is
    # present, or assumed to be un-normalized otherwise.
    labels = []
    features = []
    for tp in train_paths:
        tp_labels = np.load(os.path.join(tp, 'labels.npy'))
        labels.append(tp_labels)
        if os.path.isfile(os.path.join(tp, 'sc.joblib')):
            sc_train = joblib.load(os.path.join(tp, 'sc.joblib'))
            features.append(sc_train.inverse_transform(np.load(os.path.join(tp, 'features.npy'))))
        else:
            features.append(np.load(os.path.join(tp, 'features.npy')))
    labels = np.concatenate(labels)
    features = np.vstack(features)

    if os.path.isfile(os.path.join(test_path, 'sc.joblib')):
        sc_test = joblib.load(os.path.join(test_path, 'sc.joblib'))
        x_test = sc_test.inverse_transform(x_test)
    
    # Randomize the order of the training set so that the validation set is different each time
    data = np.hstack([features, labels.reshape(-1, 1)])
    np.random.shuffle(data)
    features = data[:,:-1]
    labels = data[:,-1:]

    y_test = np.minimum(y_test, np.ones_like(y_test))    # label water (2) as nm (1)
    labels = np.minimum(labels, np.ones_like(labels))

    # Binarize class labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels.reshape(-1, 1))
    if not args.sort:
        y_test = lb.transform(y_test.reshape(-1, 1))
        print(y_test)
    
    if args.retrain:
        # Retrain the model
        model = create_model(features.shape[1])
        history = model.fit(features, labels, batch_size=32, epochs=10)
    else:
        # Load saved model
        model = tf.keras.models.load_model(os.path.join(train_path, 'fc_model.h5'))
    print(model.summary())
    if args.validate:
        y_pred = np.round(model.predict(x_test))    # Round sigmoid output to 0 or 1
        cm = confusion_matrix(y_test, y_pred)
        if np.unique(labels).shape[0] < 3:
            print(classification_report(y_test, y_pred, digits=6))
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
        y_pred = np.rint(model.predict(x_test)).astype(int)
        y_labels = le_train.inverse_transform(y_pred)
        for l in le_train.classes_:
            # Make class directories
            os.makedirs(os.path.join(in_dir, l), exist_ok=True)
        # Move all images to the correct class directory
        for i in tqdm(range(len(fnames))):
            src = os.path.join(in_dir, fnames[i])
            dst = os.path.join(out_dir, y_labels[i], fnames[i])
            try:
                shutil.move(src, dst)
            except FileNotFoundError:
                # pass
                print('no such file '+src)
    elif args.xv:
        # 10-fold corss-validation
        kfold = KFold(n_splits=10)
        cvscores = []
        for train_index, test_index in kfold.split(features):
            model = create_model(features.shape[1])
            x_train, x_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            model.fit(x_train, y_train, batch_size=64, verbose=0, epochs=10)
            scores = model.evaluate(x_test, y_test, verbose=0)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            cvscores.append(scores[1]*100)
        print('%.3f%% +/- %.3f%%' % (np.mean(cvscores), np.std(cvscores)))
    elif args.analyze:
        # Do PCA on the features
        pca = PCA()
        pca.fit(features)
        print(np.cumsum(pca.explained_variance_ratio_))
    if args.retrain:
        # Save the retrained model
        model.save(os.path.join(train_path, 'fc_model.h5'))
