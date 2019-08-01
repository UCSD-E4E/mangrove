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

# 95%m, 97%nm site 7 recall with sparse categorical cross-entropy and 10 epochs
# 76%m, 98%nm site 8 recall w/ 256 neurons, up to ~82%m recall w/ 0.1 dropout
# presence of water as a separate class improves m recall by 1-2% on average
# goes to 83%m recall on site 8 w/ 2 dense layer-dropout pairs
# Training on site 8 is >95% for 7 and 9, but bad for the training set even w/o water

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='path to directory with output of CNN feature extractor')
    parser.add_argument('-r', '--retrain', action='store_true', help='retrain the model')
    args = parser.parse_args()
    input_path = args.input
    features = np.load(os.path.join(input_path, 'features.npy'))
    labels = np.load(os.path.join(input_path, 'labels.npy'))
    data = np.hstack([features, labels.reshape(-1, 1)])
    np.random.shuffle(data)
    features = data[:,:-1]
    labels = data[:,-1:]
    le = joblib.load(os.path.join(input_path, 'le.joblib'))
    x_test = np.load(os.path.abspath('output-site5/features.npy'))
    y_test = np.load('output-site5/labels.npy')

    oh = OneHotEncoder()
    labels = oh.fit_transform(labels.reshape(-1, 1))
    y_test = oh.transform(y_test.reshape(-1, 1))
    if args.retrain:
        inputs = tf.keras.Input(shape=(512,))
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.Dropout(rate=0.1, noise_shape=(256,))(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(rate=0.1, noise_shape=(256,))(x)
        outputs = layers.Dense(3, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(),
            metrics=['accuracy'])
        history = model.fit(features, labels, batch_size=64, epochs=10, validation_split=0.1)
    else:
        model = tf.keras.models.load_model(os.path.join(input_path, 'fc_model.h5'))
    y_pred = np.argmax(model.predict(x_test), axis=1)
    print(np.unique(y_pred))
    print(classification_report(np.argmax(y_test, axis=1), y_pred, digits=6))
    print(confusion_matrix(np.argmax(y_test, axis=1), y_pred))
    if args.retrain:
        model.save(os.path.join(input_path, 'fc_model.h5'))
