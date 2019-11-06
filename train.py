import numpy as np
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical

from keras.models import Model
from keras.utils import plot_model
from keras.layers import Input, Dense, BatchNormalization, LSTM, Activation, Dropout

from keras.optimizers import Adam

from PIL import Image
from matplotlib import pyplot as plt


class SpeechCommandsSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        #         return np.ceil(len(self.x) / float(self.batch_size))
        return 1024

    def __getitem__(self, idx):
        #         batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        #         batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        #         return np.array([
        #             resize(imread(file_name), (200, 200))
        #                for file_name in batch_x]), np.array(batch_y)

        return np.random.randn(self.batch_size, 128, 44), \
               to_categorical(np.random.randint(12, size=self.batch_size), num_classes=12)


def get_model():
    input_op = Input(shape=(128, 44))

    # BATCH_NORM
    x = BatchNormalization()(input_op)

    # LSTM
    x = LSTM(256)(x)

    # BATCH_NORM
    x = BatchNormalization()(x)

    # DENSE
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    # DENSE
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    # DENSE
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    output_op = Dense(12)(x)

    return Model(inputs=input_op, outputs=output_op)


def main():
    model = get_model()

    # plot_model(model, to_file='model.png', show_shapes=True)
    # im = Image.open('model.png')
    # plt.figure(figsize=(10, 40))
    # plt.imshow(im)

    model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    batch_size = 16
    generator = SpeechCommandsSequence([], [], batch_size)
    model.fit_generator(generator=generator, steps_per_epoch=len(generator))


if __name__ == '__main__':
    main()
