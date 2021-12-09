"""Custom TFX example gen executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tfx.types import artifact_utils
from tfx.utils import json_utils
import pandas as pd
import os
import uuid
import numpy as np
import tensorflow as tf
from tfx.components.base import base_executor
import json


def df_to_tfrecords(df, folder, filename, compression_type='GZIP', compression_level=9, columns=None, max_mb=50):
  schema = get_schema(df, columns)
  tfrecords = get_tfrecords(df, schema)

  if max_mb:
    tfrecords = split_by_size(tfrecords, max_mb=max_mb)
  else:
    tfrecords = [[tfrecord for tfrecord in tfrecords]]

  write_tfrecords(tfrecords, folder, filename,
                  compression_type=compression_type,
                  compression_level=compression_level)


def write_tfrecords(tfrecords, folder, filename, compression_type=None, compression_level=9):
  ### Function adapted from lib pandas_tfrecords, didn't use the original due to tf version conflict.
  compression_ext = '.gz' if compression_type else ''
  os.makedirs(folder, exist_ok=True)
  uid = str(uuid.uuid4())
  opts = {}
  filename = '.'.join(filename.split('.')[0:-1])

  if compression_type:
    opts['options'] = tf.io.TFRecordOptions(
      compression_type=compression_type,
      compression_level=compression_level,
    )

  for idx, chunk in enumerate(tfrecords):
    file_path = f'{folder}/{filename}{compression_ext}'

    with tf.io.TFRecordWriter(file_path, **opts) as writer:
      for item in chunk:
        writer.write(item.SerializeToString())


def get_tfrecords(df, schema):
  for _, row in df.iterrows():
    features = {}
    feature_lists = {}

    for col, val in row.items():
      f = schema[col](val)

      if type(f) is tf.train.FeatureList:
        feature_lists[col] = f

      if type(f) is tf.train.Feature:
        features[col] = f

    context = tf.train.Features(feature=features)
    if feature_lists:
      ex = tf.train.SequenceExample(
        context=context,
        feature_lists=tf.train.FeatureLists(feature_list=feature_lists))
    else:
      ex = tf.train.Example(features=context)
    yield ex


def get_schema(df, columns=None):
  schema = {}

  for col, val in df.iloc[0].to_dict().items():
    if columns and col not in columns:
      continue

    if isinstance(val, (list, np.ndarray)):
      schema[col] = (lambda f: lambda x: \
        tf.train.FeatureList(feature=[f(i) for i in x]))(_get_feature_func(val[0]))
    else:
      schema[col] = (lambda f: lambda x: f(x))(_get_feature_func(val))
  return schema


def split_by_size(tfrecords, max_mb=50):
  max_size = max_mb * 1024 * 1024
  cur_size = 0
  item = []

  for row in tfrecords:
    if cur_size + row.ByteSize() > max_size:
      yield item
      item = []
      cur_size = 0

    item.append(row)
    cur_size = cur_size + row.ByteSize()
  yield item


def _get_feature_func(val):
  if isinstance(val, (bytes, str)):
    return _bytes_feature

  if isinstance(val, (int, np.integer)):
    return _int64_feature

  if isinstance(val, (float, np.floating)):
    return _float_feature

  raise Exception(f'Unsupported type {type(val)!r}')


def _bytes_feature(value):
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()
  if isinstance(value, str):
    value = str.encode(value)
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class Executor(base_executor.BaseExecutor):

  def Do(self, input_dict, output_dict, exec_properties):
    self._log_startup(input_dict, output_dict, exec_properties)
    with open(json_utils.loads(exec_properties['feature_description']), "rb") as read_file:
      feature_description = json.load(read_file)
    output_uri = artifact_utils.get_single_uri(output_dict['examples'])
    input_uri = exec_properties['input_path']
    for file in os.listdir(input_uri):
      df = pd.read_csv(input_uri + '/' + file)
      for k, v in feature_description.items():
        if k in df.columns:
          df[k] = df[k].astype(v)

      df_to_tfrecords(df, output_uri, file)
