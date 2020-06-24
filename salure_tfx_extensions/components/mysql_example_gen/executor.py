"""Custom MySQL Component Executor"""

from typing import Any, Dict, List, NamedTuple, Text, Tuple, Iterable
import datetime

import six
import apache_beam as beam
import tensorflow as tf
import pymysql
# from six import with_metaclass

from google.protobuf import json_format
from tfx import types
from tfx.components.example_gen import base_example_gen_executor
from tfx.proto import example_gen_pb2
from salure_tfx_extensions.proto import mysql_config_pb2
from tfx.components.example_gen.utils import dict_to_example
# from beam_nuggets.io import relational_db

_DEFAULT_ENCODING = 'utf-8'


def _deserialize_conn_config(conn_config: mysql_config_pb2.MySQLConnConfig) -> pymysql.Connection:
    params = {}

    if conn_config.host != '':
        params['host'] = conn_config.host
    if conn_config.user != '':
        params['user'] = conn_config.user
    if conn_config.password != '':
        params['password'] = conn_config.password
    if conn_config.database != '':
        params['database'] = conn_config.database
    if conn_config.port != 0:
        params['port'] = conn_config.port

    return pymysql.connect(**params)

# def _row_to_example(row: Dict[Text, Tuple[int, Any]]) -> tf.train.Example:
#     # TODO: Check what the cursor description output types are
#     """Convert DB result row to tf example."""
#     feature = {}
#
#     # For data type values from pymysql:
#     # https://github.com/PyMySQL/PyMySQL/blob/37eba60439039eff17b32ef1a63b45c25ea28cec/pymysql/constants/FIELD_TYPE.py
#     print('running _row_to_example')
#     print(row)
#     for key, (data_type, value) in row.items():
#         # TODO: remove print
#         if value is None:
#             feature[key] = tf.train.Feature()
#         # TINY, SHORT, LONG, LONGLONG, INT24
#         elif data_type in {1, 2, 3, 8, 9}:
#             feature[key] = tf.train.Feature(
#                 int64_list=tf.train.Int64List(value=[value]))
#         # DECIMAL, FLOAT, DOUBLE, NEWDECIMAL
#         elif data_type in {0, 4, 5, 246}:
#             feature[key] = tf.train.Feature(
#                 float_list=tf.train.FloatList(value=[value]))
#         # VARCHAR, VAR_STRING, STRING
#         elif data_type in {15, 253, 254}:
#             feature[key] = tf.train.Feature(
#                 bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(value)]))
#         # TODO: Support date data types
#         elif data_type in {'timestamp'}:
#             value = int(datetime.datetime.fromisoformat(value).timestamp())
#             feature[key] = tf.train.Feature(
#                 int64_list=tf.train.Int64List(value=[value]))
#         else:
#             # TODO: support more types
#             raise RuntimeError(
#                 'Column type {} is not supported.'.format(data_type))
#
#     return tf.train.Example(features=tf.train.Features(feature=feature))


@beam.typehints.with_input_types(Text)
@beam.typehints.with_output_types(beam.typehints.Iterable[Tuple[Text, Text, Any]])
class _ReadMySQLDoFn(beam.DoFn):

    def __init__(self,
                 mysql_config: mysql_config_pb2.MySQLConnConfig):
        super(_ReadMySQLDoFn, self).__init__()
        self.mysql_config = json_format.MessageToDict(mysql_config)

    def process(self, query: Text) -> Iterable[Tuple[Text, Text, Any]]:
        client = pymysql.connect(**self.mysql_config)
        cursor = client.cursor()
        cursor.execute(query)

        rows = cursor.fetchall()
        if rows:
            cols = []
            col_types = []
            # Returns a list of (column_name, column_type, None, ...)
            try:
                for metadata in cursor.description:
                    cols.append(metadata[0])
                    col_types.append(metadata[1])

                for r in rows:
                    a = {}
                    for col_name, field in zip(cols, r):
                        a[col_name] = field
                    # TODO: this not the right output (most likely, since examples arriving at transform are empty)
                    # TODO: change output to column_name -> value dict
                    # print(a)
                    yield a
            # except:
            #     raise
            finally:
                cursor.close()
                client.close()


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(tf.train.Example)
def _MySQLToExample(
        pipeline: beam.Pipeline,
        input_dict: Dict[Text, List[types.Artifact]],
        exec_properties: Dict[Text, any],
        split_pattern: Text) -> beam.pvalue.PCollection:
    conn_config = example_gen_pb2.CustomConfig()
    json_format.Parse(exec_properties['custom_config'], conn_config)
    mysql_config = mysql_config_pb2.MySQLConnConfig()
    conn_config.custom_config.Unpack(mysql_config)

    # print('Starting pipeline')

    return (pipeline
            | 'Query' >> beam.Create([split_pattern])
            | 'ReadFromDB' >> beam.ParDo(_ReadMySQLDoFn(mysql_config))
            | 'ToTFExample' >> beam.Map(dict_to_example))


class Executor(base_example_gen_executor.BaseExampleGenExecutor):
    """Generic TFX MySQL executor"""

    def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
        """Returns PTransform for MySQl to TF examples."""
        return _MySQLToExample
