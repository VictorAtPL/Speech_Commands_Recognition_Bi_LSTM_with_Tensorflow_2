from argparse import ArgumentParser
from glob import glob
from importlib import import_module

import numpy as np
import tensorflow as tf

from common import load_labels, load_pickle_file


def run_prediction(args):
    batch_size = args.batch_size

    model_class = import_module("models.{}".format(args.model)).Model()

    model = model_class.get_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    model.load_weights(args.checkpoint)

    prediction_generator, _ = \
        model_class.get_input_fn_and_steps_per_epoch('prediction', batch_size)

    results = model.predict(prediction_generator, batch_size=None)

    predicted_labels_id = np.argmax(results, axis=1)

    id_to_labels, _ = load_labels()
    predicted_labels = [id_to_labels[label_id] for label_id in predicted_labels_id]

    test_filenames = list(sorted(list(load_pickle_file('test_filenames.pickle'))))

    print("fname,label")

    for filename, predicted_label in zip(test_filenames, predicted_labels):
        print("{},{}".format(filename, predicted_label))


def main():
    parser = ArgumentParser(description='DL-MAI project #2 (RNN) prediction script.')

    available_models = [model_name.split("/")[1].split(".")[0] for model_name in glob("models/*.py")]
    parser.add_argument('model', choices=available_models)
    parser.add_argument('checkpoint', metavar="model.ckpt") # type=lambda x: is_valid_file(parser, x)
    parser.add_argument('--batch-size', default=1024, type=int)

    args = parser.parse_args()
    run_prediction(args)


if __name__ == '__main__':
    main()
