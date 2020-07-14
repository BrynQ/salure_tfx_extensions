"""Helper functions for parsing and handling tf.Examples"""

import os
import pandas as pd
import itertools
from tfx import types
from typing import Text, List, Any, Union, Tuple
import tensorflow as tf
import apache_beam as beam
import numpy as np
import pyarrow
import absl


def example_to_list(example: tf.train.Example) -> List[Union[Text, int, float]]:
    # Based on the tensorflow example.proto and tensorflow feature.proto files
    result = []
    for key in example.features.feature:
        feature_value = example.features.feature[key]
        result.append(feature_value[feature_value.WhichOneof('kind')])

    return result


def to_numpy_ndarray(matrix: List[List[Any]]) -> np.ndarray:
    return np.array(matrix)


def get_train_and_eval_uris(artifact: types.Artifact, splits: List[Text]) -> Tuple[Text, Text]:
    if not ('train' in splits and 'eval' in splits):
        raise ValueError('Missing \'train\' and \'eval\' splits in \'examples\' artifact,'
                         'got {} instead'.format(splits))
    return (os.path.join(artifact.uri, 'train'),
            os.path.join(artifact.uri, 'eval'))


class CombineFeatureLists(beam.CombineFn):
    def create_accumulator(self, *args, **kwargs):
        return []

    def add_input(self, mutable_accumulator, element, *args, **kwargs):
        return mutable_accumulator.append(element)

    def merge_accumulators(self, accumulators, *args, **kwargs):
        return [item for acc in accumulators for item in acc]

    def extract_output(self, accumulator, *args, **kwargs):
        return accumulator


class RecordBatchesToTable(beam.CombineFn):
    """Combine a pcoll of RecordBatches into a Table"""

    # TODO
    def create_accumulator(self, *args, **kwargs):
        return []

    def add_input(self, mutable_accumulator, element, *args, **kwargs):
        absl.logging.info(element)
        mutable_accumulator.append(element)
        return mutable_accumulator

    def merge_accumulators(self, accumulators, *args, **kwargs):
        return sum(accumulators, [])
        # return [item for acc in accumulators for item in acc]
        # def none_acc_to_list(acc):
        #     if acc:
        #         return acc
        #     return []
        # accumulators = list(map(none_acc_to_list, accumulators))
        # return list(itertools.chain(*list(map(none_acc_to_list, accumulators))))

        # for acc in accumulators[1:]:
        #     accumulators[0].extend(acc)
        # return accumulators[0]

    def extract_output(self, accumulator, *args, **kwargs):
        absl.logging.info('accumulator: {}'.format(accumulator))
        return pyarrow.Table.from_batches(accumulator)


def from_tfrecords(file_paths, schema, compression_type='GZIP'):
    if not isinstance(file_paths, list):
        # For the case there is a wildcard in the path, like: '*'
        file_paths = tf.data.Dataset.list_files(file_paths)

    dataset = tf.data.TFRecordDataset(
        file_paths, compression_type=compression_type)

    features = schema_to_dict(schema)

    return dataset.map(lambda x: tf.io.parse_single_example(
        x, features=features))


def schema_to_dict(schema):
    features = {}

    for col, col_type in schema.items():
        features[col] = tf.io.FixedLenFeature(
            (), _get_feature_type(type=col_type))

    return features


def to_pandas(tfrecords):
    # TODO: Could use a performance increase
    df = None

    for row in tfrecords:
        if df is None:
            df = pd.DataFrame(columns=row.keys())

        df.append(row, ignore_index=True)

    return df


def _get_feature_type(type):
    return {
        int: tf.int64,
        float: tf.float32,
        str: tf.string,
        bytes: tf.string,
    }[type]
