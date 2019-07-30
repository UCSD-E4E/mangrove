import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import os, argparse, joblib
import matplotlib.pyplot as plt

# 95%m, 97%nm site 7 recall with sparse categorical cross-entropy and 10 epochs
# 76%m, 98%nm site 8 recall w/ 256 neurons
# 84%m, 95%nm site 8 recall w. 310 neurons
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='path to directory with output of CNN feature extractor')
    args = parser.parse_args()
    input_path = args.input
    features = np.load(os.path.join(input_path, 'features.npy'))
    labels = np.load(os.path.join(input_path, 'labels.npy'))
    data = np.hstack([features, labels.reshape(-1, 1)])
    np.random.shuffle(data)
    features = data[:,:-1]
    labels = data[:,-1:]
    le = joblib.load(os.path.join(input_path, 'le.joblib'))
    x_test = np.load(os.path.abspath('output-site8/features.npy'))
    y_test = np.load('output-site8/labels.npy')
    oh = OneHotEncoder()
    labels = oh.fit_transform(labels.reshape(-1, 1))
    y_test = oh.transform(y_test.reshape(-1, 1))

    inputs = tf.keras.Input(shape=(512,))
    x = layers.Dense(256, activation='relu')(inputs)
    outputs = layers.Dense(3, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(),
            metrics=['accuracy'] )
    history = model.fit(features, labels, batch_size=64, epochs=20)
    y_pred = np.argmax(model.predict(x_test), axis=1)
    print(classification_report(np.argmax(y_test, axis=1), y_pred, digits=6))