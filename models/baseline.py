import tensorflow as tf

from AbstractModel import AbstractModel
from common import get_input_fn_and_steps_per_epoch, load_sets_count, mel_spectrogram_unlabeled_parser, \
    mel_spectrogram_labeled_parser
from constants import TFRECORDS_SAVE_PATH


class Model(AbstractModel):

    def get_model(self):
        input_op = tf.keras.Input(shape=(128, 44))

        layers = tf.keras.layers
        # BATCH_NORM
        x = layers.BatchNormalization()(input_op)

        # LSTM
        # https://github.com/tensorflow/tensorflow/issues/30263
        x = layers.LSTM(256, activation='sigmoid')(x)

        # BATCH_NORM
        x = layers.BatchNormalization()(x)

        # DENSE
        x = layers.Dense(256)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)

        # DENSE
        x = layers.Dense(128)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)

        # DENSE
        x = layers.Dense(64)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)

        output_op = layers.Dense(12)(x)

        return tf.keras.Model(inputs=input_op, outputs=output_op)

    def get_input_fn_and_steps_per_epoch(self, set_name, batch_size=None):
        sets_count = load_sets_count()

        parser_fn = mel_spectrogram_labeled_parser

        if 'prediction' in set_name:
            parser_fn = mel_spectrogram_unlabeled_parser

        return get_input_fn_and_steps_per_epoch(set_name, parser_fn, TFRECORDS_SAVE_PATH,
                                                batch_size, sets_count)

