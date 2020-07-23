"""Helper functions for parsing and handling tf.Examples"""

import os
import pandas as pd
import itertools
from tfx import types
from typing import Text, List, Any, Union, Tuple, Dict
import tensorflow as tf
import apache_beam as beam
import numpy as np
import pyarrow
import absl
from tensorflow_metadata.proto.v0 import schema_pb2
import tensorflow_datasets as tfds
from google.protobuf import json_format
import base64
import six
from salure_tfx_extensions import constants

_DEFAULT_ENCODING = 'utf-8'


def dict_to_example(instance: Dict[Text, Any]) -> tf.train.Example:
    """Converts dict to tf example.
    From tfx.components.example_gen.utils.
    To preserve compatibility, as there don't seem to be compatibility guarantees for that module"""
    feature = {}
    for key, value in instance.items():
        if value is None:
            feature[key] = tf.train.Feature()
            return tf.train.Example(features=tf.train.Features(feature=feature))

        # Check if is numpy array, to parse to list
        if isinstance(value, np.ndarray):
            if value.ndim > 1:
                raise ValueError('numpy ndarrays of more than one dimension are not supported')
            value = value.tolist()

        if isinstance(value, six.integer_types):
            feature[key] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[value]))
        elif isinstance(value, float):
            feature[key] = tf.train.Feature(
                float_list=tf.train.FloatList(value=[value]))
        elif isinstance(value, six.text_type) or isinstance(value, str):
            feature[key] = tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[value.encode(_DEFAULT_ENCODING)]))
        elif isinstance(value, bytes):
            feature[key] = tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[value]))
        elif isinstance(value, list):
            if not value:
                feature[key] = tf.train.Feature()
            elif isinstance(value[0], six.integer_types):
                feature[key] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=value))
            elif isinstance(value[0], float):
                feature[key] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=value))
            elif isinstance(value[0], six.text_type) or isinstance(value[0], str):
                feature[key] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[v.encode(_DEFAULT_ENCODING) for v in value]))
            elif isinstance(value[0], bytes):
                feature[key] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=value))
            else:
                raise RuntimeError('Column type `list of {}` is not supported.'.format(
                    type(value[0])))
        else:
            raise RuntimeError('Column type {} is not supported.'.format(type(value)))
    return tf.train.Example(features=tf.train.Features(feature=feature))


# def example_to_list(example: tf.train.Example) -> List[Union[Text, int, float]]:
#     # Based on the tensorflow example.proto and tensorflow feature.proto files
#     result = []
#     for key in example.features.feature:
#         feature_value = example.features.feature[key]
#         result.append(feature_value[feature_value.WhichOneof('kind')])
#
#     return result


# def to_numpy_ndarray(matrix: List[List[Any]]) -> np.ndarray:
#     return np.array(matrix)


# def get_train_and_eval_uris(artifact: types.Artifact, splits: List[Text]) -> Tuple[Text, Text]:
#     if not ('train' in splits and 'eval' in splits):
#         raise ValueError('Missing \'train\' and \'eval\' splits in \'examples\' artifact,'
#                          'got {} instead'.format(splits))
#     return (os.path.join(artifact.uri, 'train'),
#             os.path.join(artifact.uri, 'eval'))


# class CombineFeatureLists(beam.CombineFn):
#     def create_accumulator(self, *args, **kwargs):
#         return []
#
#     def add_input(self, mutable_accumulator, element, *args, **kwargs):
#         return mutable_accumulator.append(element)
#
#     def merge_accumulators(self, accumulators, *args, **kwargs):
#         return [item for acc in accumulators for item in acc]
#
#     def extract_output(self, accumulator, *args, **kwargs):
#         return accumulator


# class RecordBatchesToTable(beam.CombineFn):
#     """Combine a pcoll of RecordBatches into a Table"""
#
#     # TODO
#     def create_accumulator(self, *args, **kwargs):
#         return []
#
#     def add_input(self, mutable_accumulator, element, *args, **kwargs):
#         absl.logging.info(element)
#         mutable_accumulator.append(element)
#         return mutable_accumulator
#
#     def merge_accumulators(self, accumulators, *args, **kwargs):
#         return sum(accumulators, [])
#         # return [item for acc in accumulators for item in acc]
#         # def none_acc_to_list(acc):
#         #     if acc:
#         #         return acc
#         #     return []
#         # accumulators = list(map(none_acc_to_list, accumulators))
#         # return list(itertools.chain(*list(map(none_acc_to_list, accumulators))))
#
#         # for acc in accumulators[1:]:
#         #     accumulators[0].extend(acc)
#         # return accumulators[0]
#
#     def extract_output(self, accumulator, *args, **kwargs):
#         absl.logging.info('accumulator: {}'.format(accumulator))
#         return pyarrow.Table.from_batches(accumulator)

#
# def from_tfrecords(file_paths, schema, compression_type='GZIP'):
#     if not isinstance(file_paths, list):
#         # For the case there is a wildcard in the path, like: '*'
#         file_paths = tf.data.Dataset.list_files(file_paths)
#
#     dataset = tf.data.TFRecordDataset(
#         file_paths, compression_type=compression_type)
#
#     return tfds.as_numpy(dataset)
#
#     # feature_types = extract_schema_features(schema)
#
#     # Research whether we need default values
#     # features = {k: tf.io.FixedLenFeature((), _to_tf_dtypes(v), default_value=_default_value_for_type(v))
#     #             for k, v in feature_types.items()}
#     #
#     # return dataset.map(lambda x: tf.io.parse_single_example(
#     #     x, features=features))


# def _default_value_for_type(type):
#     if type in [schema_pb2.FeatureType.BYTES, 'BYTES']:
#         # return tf.strings.as_string('')
#         return ''
#     if type in [schema_pb2.FeatureType.INT, 'INT']:
#         return 0
#     if type in [schema_pb2.FeatureType.FLOAT, 'FLOAT']:
#         return 0.0
#     # return tf.strings.as_string('')
#     return ''


# def _to_tf_dtypes(type):
#     if type in [schema_pb2.FeatureType.BYTES, 'BYTES']:
#         return tf.dtypes.string
#     if type in [schema_pb2.FeatureType.INT, 'INT']:
#         return tf.dtypes.int64
#     if type in [schema_pb2.FeatureType.FLOAT, 'FLOAT']:
#         return tf.dtypes.float32
#     return tf.dtypes.string


# def extract_schema_features(schema):
#     features = {}
#
#     schema_dict = json_format.MessageToDict(schema, preserving_proto_field_name=True)
#
#     for item in schema_dict['feature']:
#         features[item['name']] = _get_feature_type(_to_tf_dtypes(item['type']))
#
#     return features

#
# def to_pandas(tfrecords, schema):
#     # TODO: Could use a performance increase
#     df = None
#     schema_dict = json_format.MessageToDict(schema)
#
#     columns = [item['name'] for item in schema_dict['feature']]
#     for row in tfrecords:
#         if df is None:
#             df = pd.DataFrame(columns=columns)
#
#         df.append(pd.DataFrame(row, columns=columns), ignore_index=True)
#
#     return df


# def _get_feature_type(type):
#     # return {
#     #     int: tf.int64,
#     #     float: tf.float32,
#     #     str: tf.string,
#     #     bytes: tf.string,
#     # }[type]
#
#     if type in [schema_pb2.FeatureType.BYTES, 'BYTES']:
#         return tf.string
#     if type in [schema_pb2.FeatureType.INT, 'INT']:
#         return tf.int64
#     if type in [schema_pb2.FeatureType.FLOAT, 'FLOAT']:
#         return tf.float32
#     return tf.string


# def parse_feature_dict(feature):
#     # Dictionary based on tf.Example proto
#     feature_list = feature['features']['feature']
#
#     result = {}
#     # bytesList, floatList, int64List
#     for key in feature_list.keys():
#         if 'bytesList' in feature_list[key]:
#             result[key] = feature_list[key]['bytesList']['value']
#         elif 'floatList' in feature_list[key]:
#             result[key] = list(map(float, feature_list[key]['floatList']['value']))
#         elif 'int64List' in feature_list[key]:
#             result[key] = list(map(int, feature_list[key]['int64List']['value']))
#         else:
#             result[key] = [None]
#
#     return result


# def dataframe_from_feature_dicts(features, schema):
#     schema_dict = json_format.MessageToDict(schema, preserving_proto_field_name=True)
#
#     # Dictionary based on tf.Example proto
#     features_list = list(map(lambda x: x['features']['feature'], features))
#     columns = [item['name'] for item in schema_dict['feature']]
#
#     result = {}
#     # bytesList, floatList, int64List
#     item = features_list[0]
#     for key in item.keys():
#         if 'bytesList' in item[key]:
#             result[key] = list(map(base64.b64decode, item[key]['bytesList']['value']))
#         elif 'floatList' in item[key]:
#             result[key] = list(map(float, item[key]['floatList']['value']))
#         elif 'int64List' in item[key]:
#             result[key] = list(map(int, item[key]['int64List']['value']))
#         else:
#             result[key] = [None]
#
#     for item in features_list[1:]:
#         for key in item.keys():
#             if 'bytesList' in item[key]:
#                 result[key].extend(list(map(base64.b64decode, item[key]['bytesList']['value'])))
#             elif 'floatList' in item[key]:
#                 result[key].extend(list(map(float, item[key]['floatList']['value'])))
#             elif 'int64List' in item[key]:
#                 result[key].extend(list(map(int, item[key]['int64List']['value'])))
#             else:
#                 result[key].append(None)
#
#     return pd.DataFrame(result, columns=columns)


@beam.ptransform_fn
@beam.typehints.with_input_types(Union[tf.train.Example,
                                       tf.train.SequenceExample, bytes])
@beam.typehints.with_output_types(beam.pvalue.PDone)
def WriteSplit(example_split: beam.pvalue.PCollection,
               output_split_path: Text) -> beam.pvalue.PDone:
    """Shuffles and writes output split as serialized records in TFRecord.
    From tfx.components.example_gen, but it lacks compatibility guarantees"""

    def _MaybeSerialize(x):
        if isinstance(x, (tf.train.Example, tf.train.SequenceExample)):
            return x.SerializeToString()
        return x

    return (example_split
            | 'MaybeSerialize' >> beam.Map(_MaybeSerialize)
            | 'Shuffle' >> beam.transforms.Reshuffle()
            | 'Write' >> beam.io.WriteToTFRecord(
                os.path.join(output_split_path, constants.DEFAULT_TFRECORD_FILE_NAME),
                file_name_suffix='.gz'))
