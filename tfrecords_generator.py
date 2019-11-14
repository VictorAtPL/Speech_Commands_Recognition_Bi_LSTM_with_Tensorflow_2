import glob
import hashlib
import multiprocessing
import os
import pickle
import random
import traceback
from argparse import ArgumentParser
from collections import defaultdict, Counter
from functools import partial

import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
from tqdm import tqdm

from common import load_pickle_file
from constants import TFRECORDS_FORMAT_PATTERN, KAGGLE_LABELS, TRAIN_AUDIO_PATH, TFRECORDS_SAVE_PATH, TEST_AUDIO_PATH


def get_train_examples_sets():
    examples_sets = defaultdict(set)

    with open("train/testing_list.txt") as f:
        for line in f:
            examples_sets["test"].add(line.strip())

    with open("train/validation_list.txt") as f:
        for line in f:
            examples_sets["validation"].add(line.strip())

    for filename in glob.glob("train/**/*_nohash_*.wav", recursive=True):
        if 'silence' in filename:
            continue
        examples_sets["train"].add(filename.replace("train/audio/", ""))

    examples_sets["train"] = examples_sets["train"] - examples_sets["validation"] - examples_sets["test"]

    return examples_sets


def get_test_examples_sets():
    examples_sets = {
        'prediction': set()
    }

    for filename in glob.glob("test/audio/clip_*.wav", recursive=True):
        examples_sets["prediction"].add(filename.replace("test/audio/", ""))

    return examples_sets


def get_class_count_per_sets(examples_sets):
    series_dict = {}
    for set_name, example_set in examples_sets.items():
        set_counter = Counter()
        for filename in list(example_set):
            label = filename.split("/")[0]
            set_counter[label] += 1

        set_counter_dict = dict(set_counter)
        series_dict[set_name] = pd.Series(data=list(set_counter_dict.values()), index=list(set_counter_dict.keys()))

    return pd.DataFrame(series_dict)


def get_percentage_sets_split(df_examples):
    return get_sum_sets_split(df_examples) / df_examples.sum().sum()


def get_sum_sets_split(df_examples):
    return df_examples.sum(axis=0)


def generate_silence():
    if not os.path.isdir("train/audio/silence"):
        os.mkdir("train/audio/silence")
        # shutil.rmtree("train/audio/silence", ignore_errors=True)
    else:
        return

    for filepath in tqdm(glob.glob("train/audio/_background_noise_/*.wav")):
        sample_rate, samples = scipy.io.wavfile.read(str(filepath))
        sections = int(samples.shape[0] / sample_rate)

        # If we used code below, there will be 4x less silence audios than other classes
        # chunks = np.split(samples[:sections * sample_rate], sections)

        chunks = [samples[start_index:start_index + sample_rate] for start_index in np.random.randint(samples.shape[0] - sample_rate, size=sections * 6)]

        filename = filepath.split("/")[len(filepath.split("/")) - 1]
        noise_name = filename.split(".")[0]
        noise_hash = hashlib.sha256(noise_name.encode()).hexdigest()[:8]

        for i, chunk in tqdm(enumerate(chunks), position=1, total=len(chunks)):
            scipy.io.wavfile.write("train/audio/silence/{}_nohash_{}.wav".format(noise_hash, i), sample_rate, chunk)


def get_silence_subpaths():
    return ['/'.join(path.split('/')[2:]) for path in glob.glob("train/audio/silence/*_nohash_*.wav")]


def split_silence_per_sets(percentage_sets_split, silence_subpaths):
    silence_waves_count = len(silence_subpaths)

    silence_subpaths_shuffled = random.sample(silence_subpaths, silence_waves_count)

    silence_subpaths_for_test_count = int(percentage_sets_split["test"] * silence_waves_count)
    silence_subpaths_for_validation_count = int(percentage_sets_split["validation"] * silence_waves_count)
    silence_subpaths_for_train_count = silence_waves_count - silence_subpaths_for_test_count - silence_subpaths_for_validation_count

    silence_subpaths_for_test = silence_subpaths_shuffled[:silence_subpaths_for_test_count]
    silence_subpaths_for_validation = silence_subpaths_shuffled[
                                      silence_subpaths_for_test_count:silence_subpaths_for_test_count + silence_subpaths_for_validation_count]
    silence_subpaths_for_train = silence_subpaths_shuffled[
                                 silence_subpaths_for_test_count + silence_subpaths_for_validation_count:]

    assert len(silence_subpaths_for_test) + len(silence_subpaths_for_validation) + len(
        silence_subpaths_for_train) == len(silence_subpaths)

    return {
        "test": silence_subpaths_for_test,
        "validation": silence_subpaths_for_validation,
        "train": silence_subpaths_for_train
    }


def get_labels_dicts(labels):
    return labels, {label: i for i, label in enumerate(labels)}


def generate_tfrecords_for_set(output_path, set_name, file_pattern, filenames, audio_path, id_to_labels,
                               labels_to_id, number_of_shards=1, cpu_count=1):
    filenames_chunks = np.array_split(list(filenames), number_of_shards)
    part_files_arr = zip(range(number_of_shards), filenames_chunks)

    try:
        # pool = multiprocessing.Pool(min(cpu_count, number_of_shards))
        #
        # results = [pool.apply_async(partial(
        #         convert_wavs_to_tfrecords,
        #         part_files=part_files,
        #         output_path=output_path,
        #         set_name=set_name,
        #         file_pattern=file_pattern,
        #         audio_path=audio_path,
        #         id_to_labels=id_to_labels,
        #         labels_to_id=labels_to_id,
        #         number_of_shards=number_of_shards),
        #         ()
        #     ) for part_files in part_files_arr]
        #
        # [p.get() for p in results]
        #
        # pool.close()

        [convert_wavs_to_tfrecords(
                part_files=part_files,
                output_path=output_path,
                set_name=set_name,
                file_pattern=file_pattern,
                audio_path=audio_path,
                id_to_labels=id_to_labels,
                labels_to_id=labels_to_id,
                number_of_shards=number_of_shards) for part_files in tqdm(part_files_arr, total=number_of_shards)]

    except Exception as e:
        traceback.print_exc()


def convert_wavs_to_tfrecords(part_files, output_path, set_name, file_pattern, audio_path, id_to_labels,
                              labels_to_id, number_of_shards=1):
    import librosa

    part_no, part_files = part_files
    try:
        with tf.io.TFRecordWriter(
                os.path.join(output_path, file_pattern.format(set_name, part_no + 1, number_of_shards))) as writer:
            # for filename in tqdm(part_files, position = part_no):
            for filename in tqdm(part_files, position=1):
                if 'prediction' not in set_name:
                    label = filename.split("/")[0]

                    if label not in id_to_labels:
                        label = "unknown"
                    label_id = labels_to_id[label]
                else:
                    label_id = -1

                sample_rate, samples = scipy.io.wavfile.read(os.path.join(audio_path, filename))
                samples = samples.astype(float)
                # samples = np.array(samples)
                # if samples.shape[0] < sample_rate:
                #     to_pad = sample_rate - samples.shape[0]
                #     samples = np.pad(samples, pad_width=[(0, to_pad)], mode="edge")
                # elif samples.shape[0] > sample_rate:
                #     samples = samples[:sample_rate]
                #
                # assert samples.shape == (sample_rate, )
                #
                # example = tf.train.Example(features=tf.train.Features(
                #     feature={'samples': tf.train.Feature(float_list=tf.train.FloatList(value=samples.flatten())),
                #              'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_id]))}))

                S = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=128)
                log_S = librosa.power_to_db(S, ref=np.max).astype(np.float32)

                if log_S.shape != (128, 44):
                    to_pad = 44 - log_S.shape[1]
                    log_S = np.pad(log_S, pad_width=[(0, 0), (0, to_pad)], mode="edge")

                assert log_S.shape == (128, 44)

                example = tf.train.Example(features=tf.train.Features(
                    feature={'samples': tf.train.Feature(float_list=tf.train.FloatList(value=log_S.flatten())),
                             'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_id]))}))

                writer.write(example.SerializeToString())

            writer.flush()
    except Exception as e:
        traceback.print_exc()
    return 1


def generate_tfrecords_for_dataset(path, examples_sets, audio_path, id_to_labels, labels_to_id):
    if not tf.io.gfile.exists(path):
        tf.io.gfile.makedirs(path)

    for set_name, example_set in examples_sets.items():
        for filepath in glob.glob(os.path.join(path, '{}*.tfrecords'.format(set_name))):
            tf.io.gfile.remove(filepath)

        if 'prediction' in set_name:
            example_set = sorted(example_set)

        generate_tfrecords_for_set(path, set_name, TFRECORDS_FORMAT_PATTERN, example_set, audio_path, id_to_labels, labels_to_id,
                                   number_of_shards=len(example_set) // 4500)


def write_labels(path, id_to_labels, labels_to_id):
    with tf.io.gfile.GFile(os.path.join(path, "id_to_labels.pickle"), mode='wb') as f:
        pickle.dump(id_to_labels, f, protocol=pickle.HIGHEST_PROTOCOL)

    with tf.io.gfile.GFile(os.path.join(path, "labels_to_id.pickle"), mode='wb') as f:
        pickle.dump(labels_to_id, f, protocol=pickle.HIGHEST_PROTOCOL)


def write_sets_length(path, merged_examples_sets):
    examples_sets_class_count = get_class_count_per_sets(merged_examples_sets)
    sum_sets_split = get_sum_sets_split(examples_sets_class_count).to_dict()

    if tf.io.gfile.exists(os.path.join(path, "sets_count.pickle")):
        local_sum_sets_split = load_pickle_file("sets_count.pickle")

        for set_name, count in local_sum_sets_split.items():
            if set_name not in sum_sets_split.keys():
                sum_sets_split[set_name] = count

    with tf.io.gfile.GFile(os.path.join(path, "sets_count.pickle"), mode='wb') as f:
        pickle.dump(sum_sets_split, f, protocol=pickle.HIGHEST_PROTOCOL)


def generate_tfrecords_train():
    np.random.seed(0)

    examples_sets = get_train_examples_sets()

    examples_sets_class_count = get_class_count_per_sets(examples_sets)
    percentage_sets_split = get_percentage_sets_split(examples_sets_class_count)

    generate_silence()
    silence_subpaths = get_silence_subpaths()

    silence_examples_sets = split_silence_per_sets(percentage_sets_split, silence_subpaths)

    merged_examples_sets = {set_name: list(examples_sets[set_name]) + silence_examples_sets[set_name] for set_name in examples_sets.keys()}
    # merged_examples_sets_class_count = get_class_count_per_sets(merged_examples_sets)


    # df_unknown = pd.DataFrame((merged_examples_sets_class_count.sum() - merged_examples_sets_class_count.loc[kaggle_labels].sum()).rename("unknown")).transpose()
    # merged_examples_sets_class_count = pd.concat([merged_examples_sets_class_count, df_unknown])
    # merged_examples_sets_class_count = merged_examples_sets_class_count.loc[kaggle_labels]

    id_to_labels, labels_to_id = get_labels_dicts(KAGGLE_LABELS)

    generate_tfrecords_for_dataset(TFRECORDS_SAVE_PATH, merged_examples_sets, TRAIN_AUDIO_PATH, id_to_labels, labels_to_id)
    write_sets_length(os.path.join(TFRECORDS_SAVE_PATH, '../'), merged_examples_sets)


def write_test_filenames(path, examples_sets):
    with tf.io.gfile.GFile(os.path.join(path, "test_filenames.pickle"), mode='wb') as f:
        pickle.dump(examples_sets['prediction'], f, protocol=pickle.HIGHEST_PROTOCOL)


def generate_tfrecords_test():
    id_to_labels, labels_to_id = get_labels_dicts(KAGGLE_LABELS)

    examples_sets = get_test_examples_sets()

    generate_tfrecords_for_dataset(TFRECORDS_SAVE_PATH, examples_sets, TEST_AUDIO_PATH, id_to_labels,
                                   labels_to_id)
    write_sets_length(os.path.join(TFRECORDS_SAVE_PATH, '../'), examples_sets)
    write_test_filenames(os.path.join(TFRECORDS_SAVE_PATH, '../'), examples_sets)


def generate_labels():
    id_to_labels, labels_to_id = get_labels_dicts(KAGGLE_LABELS)
    write_labels(os.path.join(TFRECORDS_SAVE_PATH, '../'), id_to_labels, labels_to_id)


def main():
    parser = ArgumentParser(description='DL-MAI project #2 (RNN) tf generator script.')
    parser.add_argument("type", choices=["train", "test", "labels"])

    args = parser.parse_args()
    if 'train' in args.type:
        generate_tfrecords_train()
    elif 'test' in args.type:
        generate_tfrecords_test()
    elif 'labels' in args.type:
        generate_labels()


if __name__ == '__main__':
    main()
