import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import os, argparse, joblib


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='path to directory with output of CNN feature extractor')
    args = parser.parse_args()
    input_path = args.input
    features = np.load(os.path.join(input_path, 'features.npy'))
    labels = np.load(os.path.join(input_path, 'labels.npy'))
    le = joblib.load(os.path.join(input_path, 'le.joblib'))
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=3)

    inputs = tf.keras.Input(shape=(512,))
    x = layers.Dense(256, activation='relu')(inputs)
    outputs = layers.Dense(3, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    # x_train and x_test are alread normalized and shaped correctly
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(),
            metrics=['accuracy'] )
    history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.1)
    test_scores = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])