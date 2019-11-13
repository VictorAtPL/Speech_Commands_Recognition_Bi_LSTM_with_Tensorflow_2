import math
import os
import pickle
import sys
from argparse import ArgumentParser
from datetime import datetime
from glob import glob
from importlib import import_module

import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt

from tfrecords_generator import TFRECORDS_SAVE_PATH, TFRECORDS_FORMAT_PATTERN
from tfrecords_reader import make_input_local


def load_pickle_file(filename):
    with tf.io.gfile.GFile(os.path.join(TFRECORDS_SAVE_PATH, '../', filename), mode='rb') as f:
        return pickle.load(f)


def load_labels():
    id_to_labels = load_pickle_file('id_to_labels.pickle')
    labels_to_id = load_pickle_file('labels_to_id.pickle')

    return id_to_labels, labels_to_id


def load_sets_count():
    sets_count = load_pickle_file('sets_count.pickle')

    return sets_count


def get_input_fn_and_steps_per_epoch(set_name, batch_size, sets_count):
    assert set_name in ('train', 'validation', 'test')

    generator = make_input_local(
        glob(os.path.join(TFRECORDS_SAVE_PATH, TFRECORDS_FORMAT_PATTERN.format(set_name, '*', '*'))),
        shuffle=True if set_name == 'train' else False,
        repeat=True if set_name == 'train' else False)({'batch_size': batch_size})

    steps_per_epoch = int(np.ceil(sets_count[set_name] / batch_size))

    return generator, steps_per_epoch


def run_training(args):
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # tf.config.experimental_run_functions_eagerly(True)

    batch_size = args.batch_size
    sets_count = load_sets_count()

    model_module = import_module("models.{}.model".format(args.model))

    time = datetime.now().strftime('%d%m%Y_%H%M%S')

    def make_scheduler(base_lr):
        def scheduler(epoch):
            if epoch < 10:
                return base_lr
            else:
                return base_lr * float(math.exp(0.1 * (10 - epoch)))
        return scheduler

    model_logs_dir_name = "{}-{}-{}".format(time, args.model, args.variant) if args.variant\
        else "{}-{}".format(time, args.model)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = model_module.get_model()

        if 'pydot' in sys.modules:
            tf.keras.utils.plot_model(model, to_file=args.model + '.png', show_shapes=True)
            im = Image.open(args.model + '.png')
            plt.figure(figsize=(10, 40))
            plt.imshow(im)

        callbacks = [tf.keras.callbacks.TensorBoard(log_dir=os.path.join('logs', model_logs_dir_name),
                                                    profile_batch=0,
                                                    update_freq='epoch',
                                                    write_graph=False),
                     tf.keras.callbacks.LearningRateScheduler(make_scheduler(args.base_lr)),
                     tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=5, min_delta=0.005)]

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

    train_generator, train_steps_per_epoch = get_input_fn_and_steps_per_epoch('train', batch_size, sets_count)
    validation_generator, validation_steps_per_epoch = get_input_fn_and_steps_per_epoch('validation', batch_size,
                                                                                        sets_count)

    model.fit(train_generator, steps_per_epoch=train_steps_per_epoch, epochs=args.epochs, callbacks=callbacks,
              validation_data=validation_generator, validation_steps=validation_steps_per_epoch,
              validation_freq=[1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100])


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle


def main(args):
    parser = ArgumentParser(description='MGU project #3 & DL-MAI project #2 (RNN) training script.')

    available_models = [model_name.split("/")[1] for model_name in glob("models/*/model.py")]
    parser.add_argument('model', choices=available_models)
    parser.add_argument("--variant", type=str)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=1024, type=int)
    parser.add_argument('--base-lr', default=0.005, type=float)

    if args is not None:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()

    run_training(args)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
