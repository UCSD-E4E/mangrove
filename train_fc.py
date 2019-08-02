import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
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

def remove_water(x_test, y_test, le):
    '''
    Remove vectors labeled as water from a dataset.
    '''
    if 'water' in le.classes_:
        water_index = le.transform(['water'])[0]
        return x_test[y_test!=water_index], y_test[y_test!=water_index]
    return x_test, y_test
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train directory', default='output-site8/')
    parser.add_argument('--test', help='test directory')
    parser.add_argument('-r', '--retrain', action='store_true', help='retrain the model')
    parser.add_argument('--indir', help='directory of unlabeled images')
    parser.add_argument('--outdir', help='directory to output sorted images')
    parser.add_argument('--sort', action='store_true', help='sort unlabeled images')
    parser.add_argument('-v', '--validate', action='store_true', help='test on preprocessed data')
    args = parser.parse_args()

    train_path = os.path.abspath(args.train)
    test_path = os.path.abspath(args.test)
    features = np.load(os.path.join(train_path, 'features.npy'))
    labels = np.load(os.path.join(train_path, 'labels.npy'))
    features2 = np.load('output/features.npy')
    labels2 = np.load('output/labels.npy')
    features = np.vstack([features, features2])
    labels = np.concatenate([labels, labels2])

    data = np.hstack([features, labels.reshape(-1, 1)])
    np.random.shuffle(data)
    features = data[:,:-1]
    labels = data[:,-1:]

    le_train = joblib.load(os.path.join('output/', 'le.joblib'))
    x_test = np.load(os.path.join(test_path, 'features.npy'))
    y_test = np.load(os.path.join(test_path, 'labels.npy'))
    # print(le_train.inverse_transform([0, 1, 2]))
    # print(le_test.inverse_transform([0, 1, 2]))
    # x_test, y_test = remove_water(x_test, y_test, le_test)

    oh = OneHotEncoder()
    labels = oh.fit_transform(labels.reshape(-1, 1))
    if not args.sort:
        y_test = oh.transform(y_test.reshape(-1, 1))
    if args.retrain:
        inputs = tf.keras.Input(shape=(512,))
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.Dropout(rate=0.5, noise_shape=(256,))(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(rate=0.5, noise_shape=(256,))(x)
        outputs = layers.Dense(labels.shape[1], activation='softmax')(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(),
            metrics=['accuracy'])
        history = model.fit(features, labels, batch_size=64, epochs=10, validation_split=0.25)
    else:
        model = tf.keras.models.load_model(os.path.join(train_path, 'fc_model.h5'))
    
    if args.validate:
        y_pred = np.argmax(model.predict(x_test), axis=1)
        cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
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
    if args.retrain:
        model.save(os.path.join(train_path, 'fc_model.h5'))
