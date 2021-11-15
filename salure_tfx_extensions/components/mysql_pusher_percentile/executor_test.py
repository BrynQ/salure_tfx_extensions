import os
import pymysql
import tensorflow as tf
from google.protobuf import json_format
from tfx.dsl.io import fileio
from tfx.types import standard_artifacts
from tfx.utils import proto_utils
from salure_tfx_extensions.components.mysql_pusher_percentile.executor import Executor
from salure_tfx_extensions.proto import mysql_config_pb2
import re


# noinspection PyMethodMayBeStatic
class ExecutorTest(tf.test.TestCase):

    def setUp(self):
        super(ExecutorTest, self).setUp()
        self._source_data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "testdata")
        self._output_data_dir = os.path.join(
            os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", self.get_temp_dir()),
            self._testMethodName)
        fileio.makedirs(self._output_data_dir)
        self._prediction_logs = standard_artifacts.InferenceResult()
        self._prediction_logs.uri = os.path.join(self._source_data_dir, "prediction_logs")
        self._percentile_values = standard_artifacts.Examples()
        self._percentile_values.uri = os.path.join(self._source_data_dir, "percentile_values")
        self._table_name = "ml_results"
        self._inference_id = "inference_id"
        self._input_dict = {
            'inference_result': [self._prediction_logs],
            'percentile_values': [self._percentile_values],
        }

        self._output_dict = {}
        self._exec_properties = {
            'connection_config': proto_utils.proto_to_json(mysql_config_pb2.MySQLConnConfig(
                host='10.10.0.7',
                port=3306,
                user='mlwizardxueming',
                password='the_Albaphet',
                database='sc_medux')),
            'table_name': self._table_name,
            'inference_id': self._inference_id
        }

    def removeFiles(self, directory, pattern):
        for file in os.listdir(directory):
            if re.search(pattern, file):
                os.remove(os.path.join(directory, file))

    def testPushedToDatabase(self):
        query = f"SELECT * FROM {self._table_name}"
        query = f"SELECT COUNT(*) FROM {self._table_name}"
        config = proto_utils.json_to_proto(self._exec_properties['connection_config'],
                                           mysql_config_pb2.MySQLConnConfig())
        client = pymysql.connect(**json_format.MessageToDict(config))
        with client.cursor() as cursor:
            cursor.execute(query)
            number_of_rows_before_push = cursor.fetchall()[0][0]

        self.removeFiles(self._prediction_logs.uri, "json$")
        pusher_percentile = Executor()
        pusher_percentile.Do(self._input_dict, self._output_dict, self._exec_properties)
        self.removeFiles(self._prediction_logs.uri, "json$")

        client = pymysql.connect(**json_format.MessageToDict(config))
        with client.cursor() as cursor:
            cursor.execute(query)
            number_of_rows_after_push = cursor.fetchall()[0][0]
        self.assertGreater(number_of_rows_after_push, number_of_rows_before_push)

    def testDo_NoInput(self):
        with self.assertRaisesRegex(KeyError, "inference_result"):
            pusher_percentile = Executor()
            pusher_percentile.Do(
                input_dict={},
                output_dict=self._output_dict,
                exec_properties=self._exec_properties)


if __name__ == '__main__':
    tf.test.main()
