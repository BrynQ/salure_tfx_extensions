"""Custom MySQL Component Executor"""

from typing import Any, Dict, List, NamedTuple, Text, Tuple, Iterable
import datetime
import six
import apache_beam as beam
import tensorflow as tf
import pymysql
from google.protobuf import json_format
from tfx.components.example_gen import base_example_gen_executor
from tfx.proto import example_gen_pb2
from salure_tfx_extensions.proto import mysql_config_pb2


_DEFAULT_ENCODING = 'utf-8'


def dict_to_example(instance) -> tf.train.Example:
    """Converts dict to tf example."""
    feature = {}
    for key, value in instance.items():
        if value is None:
            feature[key] = tf.train.Feature()
        elif isinstance(value, six.integer_types):
            feature[key] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[value]))
        elif isinstance(value, float):
            feature[key] = tf.train.Feature(
                float_list=tf.train.FloatList(value=[value]))
        elif isinstance(value, six.text_type) or isinstance(value, str):
            feature[key] = tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[value.encode(_DEFAULT_ENCODING)]))
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
            else:
                raise RuntimeError('Column type `list of {}` is not supported.'.format(
                    type(value[0])))
        else:
            raise RuntimeError('Column type {} is not supported.'.format(type(value)))
    return tf.train.Example(features=tf.train.Features(feature=feature))


# @beam.typehints.with_input_types(Text)
# @beam.typehints.with_output_types(beam.typehints.Iterable[Dict[Text, Any]])
class _ReadMySQLDoFn(beam.DoFn):

    def __init__(self,
                 mysql_config: mysql_config_pb2.MySQLConnConfig):
        super(_ReadMySQLDoFn, self).__init__()
        self.mysql_config = json_format.MessageToDict(mysql_config)

    def process(self, query: Text):
        client = pymysql.connect(**self.mysql_config)
        cursor = client.cursor()
        cursor.execute(query)

        rows = cursor.fetchall()
        if rows:
            cols = []
            col_types = []
            try:
                for metadata in cursor.description:
                    cols.append(metadata[0])
                    col_types.append(metadata[1])

                for r in rows:
                    a = {}
                    for col_name, field in zip(cols, r):
                        a[col_name] = field
                    yield a
            finally:
                cursor.close()
                client.close()


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(tf.train.Example)
def _MySQLToExample(
        pipeline: beam.Pipeline,
        # input_dict: Dict[Text, List[types.Artifact]],
        exec_properties: Dict[Text, any],
        split_pattern: Text) -> beam.pvalue.PCollection:
    conn_config = example_gen_pb2.CustomConfig()
    json_format.Parse(exec_properties['custom_config'], conn_config)
    mysql_config = mysql_config_pb2.MySQLConnConfig()
    conn_config.custom_config.Unpack(mysql_config)

    return (pipeline
            | 'Query' >> beam.Create([split_pattern])
            | 'ReadFromDB' >> beam.ParDo(_ReadMySQLDoFn(mysql_config))
            | 'ToTFExample' >> beam.Map(dict_to_example))


class Executor(base_example_gen_executor.BaseExampleGenExecutor):
    """Generic TFX MySQL executor"""

    def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
        """Returns PTransform for MySQl to TF examples."""
        return _MySQLToExample
