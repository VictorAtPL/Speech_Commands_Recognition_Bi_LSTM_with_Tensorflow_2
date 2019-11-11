import math
import os
import sys
from argparse import ArgumentParser
from datetime import datetime
from glob import glob
from importlib import import_module

import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt

from DefaultDistributeStrategy import DefaultDistributeStrategy
from callbacks.TensorBoard import MyTensorBoardCallback


def run_training(args):
    gpu_list = tf.config.experimental.list_physical_devices('GPU')

    if gpu_list:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpu_list:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpu_list), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # tf.config.experimental_run_functions_eagerly(True)

    batch_size = args.batch_size

    model_class = import_module("models.{}".format(args.model)).Model()

    time = datetime.now().strftime('%d%m%Y_%H%M%S')

    def make_scheduler(lr):
        def scheduler(epoch):
            if epoch < 10:
                return lr
            else:
                return lr * float(math.exp(0.1 * (10 - epoch)))
        return scheduler

    model_logs_dir_name = "{}-{}".format(time, args.model)

    strategy = tf.distribute.MirroredStrategy() if gpu_list else DefaultDistributeStrategy()
    with strategy.scope():
        model = model_class.get_model()

        if 'pydot' in sys.modules:
            tf.keras.utils.plot_model(model, to_file=args.model + '.png', show_shapes=True)
            im = Image.open(args.model + '.png')
            plt.figure(figsize=(10, 40))
            plt.imshow(im)

        callbacks = [MyTensorBoardCallback(args, log_dir=os.path.join('logs', model_logs_dir_name),
                                           profile_batch=0,
                                           update_freq='epoch',
                                           write_graph=False)]

        if args.use_lr_scheduler:
            callbacks.append(tf.keras.callbacks.LearningRateScheduler(make_scheduler(args.lr)))

        if args.use_early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',
                                                              patience=args.early_stopping_patience,
                                                              min_delta=args.early_stopping_min_delta))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

    train_generator, train_steps_per_epoch = \
        model_class.get_input_fn_and_steps_per_epoch('train', batch_size)
    validation_generator, validation_steps_per_epoch = \
        model_class.get_input_fn_and_steps_per_epoch('validation', batch_size)

    model.fit(train_generator, steps_per_epoch=train_steps_per_epoch, epochs=args.epochs, callbacks=callbacks,
              validation_data=validation_generator, validation_steps=validation_steps_per_epoch,
              validation_freq=[1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100])


def main():
    parser = ArgumentParser(description='MGU project #3 & DL-MAI project #2 (RNN) training script.')

    available_models = [model_name.split("/")[1].split(".")[0] for model_name in glob("models/*.py")]
    parser.add_argument('model', choices=available_models)
    parser.add_argument('--comment', type=str)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=1024, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--use_lr_scheduler', action='store_true')
    parser.add_argument('--use_early_stopping', action='store_true')
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.005)

    args = parser.parse_args()
    run_training(args)


if __name__ == '__main__':
    main()
