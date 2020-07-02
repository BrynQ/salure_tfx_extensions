"""Helper functions for parsing and handling tf.Examples"""

import os
from tfx import types
from typing import Text, List, Any, Union, Tuple
import tensorflow as tf
import apache_beam as beam
import numpy as np
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
