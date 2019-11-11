import math
import os
import pickle
from glob import glob

import tensorflow as tf

from constants import TFRECORDS_FORMAT_PATTERN, TFRECORDS_SAVE_PATH


def mel_spectrogram_parser(serialized_example):
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            "samples": tf.io.FixedLenFeature([128 * 44], tf.float32),
            "label": tf.io.FixedLenFeature([], tf.int64),
        })

    features["samples"] = tf.reshape(features["samples"], [128, 44])  # * (2. / 255) - 1

    return features["samples"], features["label"]


def make_input_local(filenames, parser_fn, shuffle=False, repeat=False):
    def input_fn(params):
        batch_size = params["batch_size"]

        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(parser_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.cache()

        if repeat:
            dataset = dataset.repeat()

        if shuffle:
            dataset = dataset.shuffle(buffer_size=70000, reshuffle_each_iteration=True)

        dataset = dataset.batch(batch_size)

        if repeat:
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    return input_fn


def get_input_fn_and_steps_per_epoch(set_name, parser_fn, tfrecords_path, batch_size, sets_count):
    assert set_name in ('train', 'validation', 'test')

    steps_per_epoch = math.ceil(sets_count[set_name] / batch_size)

    generator = make_input_local(
        glob(os.path.join(tfrecords_path, TFRECORDS_FORMAT_PATTERN.format(set_name, '*', '*'))),
        parser_fn,
        shuffle=True if set_name == 'train' else False,
        repeat=True if set_name == 'train' else False)({'batch_size': batch_size})

    return generator, steps_per_epoch


# def main():
#     sample_rate = 16000
#     with tf.Graph().as_default() as graph:
#         input_fn = make_input_local(glob(os.path.join(TFRECORDS_SAVE_PATH, TFRECORDS_FORMAT_PATTERN.format('*', '*', '*'))), shuffle=True, repeat=True)({"batch_size": 8})
#
#         with tf.Session() as sess:
#             init = tf.global_variables_initializer()
#             sess.run(init)
#
#             flag = True
#             i = 1
#             while flag:
#                 try:
#                     output = sess.run([input_fn])
#
#                     for i in range(len(output[0][1])):
#                         id_to_labels, _ = get_labels_dicts(KAGGLE_LABELS)
#                         label = id_to_labels[output[0][1][i].item()]
#
#                         if label in ("unknown", "silence"):
#                             continue
#
#                         sd.play(output[0][0][0], sample_rate)
#                         time.sleep(1)
#                         sd.stop()
#
#                         pass
#                         # flag = False
#                         # break
#                         # plt.figure(figsize=(12, 4))
#                         # librosa.display.specshow(output[0][0][i], sr=sample_rate, x_axis='time', y_axis='mel')
#                         # plt.title('Mel power spectrogram of ' + label)
#                         # plt.colorbar(format='%+02.0f dB')
#                         # plt.tight_layout()
#                         # i = i + 1
#                         # if i == 3:
#                         #     flag = False
#                         #     break
#
#                 except tf.errors.OutOfRangeError:
#                     break


if __name__ == '__main__':
    # main()
    pass


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


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle