import keras
from keras import backend as K
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers import Layer, Input, Dense, Conv2D, Flatten, MaxPooling2D, AveragePooling2D

# copied from Keras functional api docs
def Inception(x, num_kernels):
    tower_1 = Conv2D(num_kernels, (1, 1), padding='same', activation='relu')(x)
    tower_1 = Conv2D(num_kernels, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(num_kernels, (1, 1), padding='same', activation='relu')(x)
    tower_2 = Conv2D(num_kernels, (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    tower_3 = Conv2D(num_kernels, (1, 1), padding='same', activation='relu')(tower_3)

    output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

    return output

# copied from Keras source
class LRN2D(Layer):
    """
    This code is adapted from pylearn2.
    License at: https://github.com/lisa-lab/pylearn2/blob/master/LICENSE.txt
    """

    def __init__(self, alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
        if n % 2 == 0:
            raise NotImplementedError(
                "LRN2D only works with odd n. n provided: " + str(n))
        super(LRN2D, self).__init__(**kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n

    def get_output(self, train):
        X = self.get_input(train)
        b, ch, r, c = K.shape(X)
        half_n = self.n // 2
        input_sqr = K.square(X)
        extra_channels = K.zeros((b, ch + 2 * half_n, r, c))
        input_sqr = K.concatenate([extra_channels[:, :half_n, :, :],
                                   input_sqr,
                                   extra_channels[:, half_n + ch:, :, :]],
                                  axis=1)
        scale = self.k
        for i in range(self.n):
            scale += self.alpha * input_sqr[:, i:i + ch, :, :]
        scale = scale ** self.beta
        return X / scale

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def create_model():
    input_img = Input(shape=(15, 15, 3))
    x = Conv2D(64, kernel_size=3, activation='relu', input_shape=(15, 15, 3))(input_img)
    x = MaxPooling2D(pool_size=(3, 3), strides=None, padding='same')(x)
    # LRN Layer 
    x = LRN2D()(x)
    x = Conv2D(100, kernel_size=1, activation='relu')(x) # need to fix
    x = Conv2D(192, kernel_size=3, activation='relu')(x)
    # LRN Layer
    x = LRN2D()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=None, padding='same')(x)
    x = Inception(x, num_kernels=64)
    x = Inception(x, num_kernels=120)
    x = Inception(x, num_kernels=128)
    x = AveragePooling2D(pool_size=(3, 3), strides=None, padding='same')(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    output = Dense(2, activation='softmax', name='output')(x)

    model = Model(inputs=input_img, outputs=output)

    return model

### import data

### preprocess data

### create CNN
model = create_model()

print(model.summary())

#plot_model(model, to_file='patch_based_CNN_graph.png')
### train CNN

### evaluate
