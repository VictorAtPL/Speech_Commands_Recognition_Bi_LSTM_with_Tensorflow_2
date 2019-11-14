from argparse import ArgumentParser
from glob import glob
from importlib import import_module

import numpy as np
import tensorflow as tf

from common import load_labels, load_pickle_file


def evaluation(args):
    batch_size = args.batch_size

    model_class = import_module("models.{}".format(args.model)).Model()

    model = model_class.get_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    model.load_weights(args.checkpoint)

    test_generator, test_steps_per_epoch = \
        model_class.get_input_fn_and_steps_per_epoch('test', batch_size)

    print(model.evaluate(test_generator))
    pass


def main():
    parser = ArgumentParser(description='DL-MAI project #2 (RNN) evaluation script.')

    available_models = [model_name.split("/")[1].split(".")[0] for model_name in glob("models/*.py")]
    parser.add_argument('model', choices=available_models)
    parser.add_argument('checkpoint', metavar="model.ckpt") # type=lambda x: is_valid_file(parser, x)
    parser.add_argument('--batch-size', default=1024, type=int)

    args = parser.parse_args()
    evaluation(args)


if __name__ == '__main__':
    main()
