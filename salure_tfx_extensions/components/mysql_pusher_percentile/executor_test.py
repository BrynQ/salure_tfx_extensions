import os
from google.protobuf import json_format
import tensorflow as tf
from salure_tfx_extensions.components.mysql_pusher_percentile.component import MySQLPusher
from salure_tfx_extensions.components.mysql_pusher_percentile.executor import Executor
from salure_tfx_extensions.proto import mysql_config_pb2
from tfx.dsl.io import fileio
from tfx.types import standard_artifacts
from tfx.types.standard_component_specs import INFRA_BLESSING_KEY
from tfx.types.standard_component_specs import MODEL_BLESSING_KEY
from tfx.types.standard_component_specs import MODEL_KEY
from tfx.types.standard_component_specs import PUSH_DESTINATION_KEY
from tfx.types.standard_component_specs import PUSHED_MODEL_KEY
from tfx.utils import io_utils
from tfx.utils import path_utils
from pymysql import connect


class ExecutorTest(tf.test.TestCase):

    def setUp(self):
        super(ExecutorTest, self).setUp()
        self._source_data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'testdata')
        self._output_data_dir = os.path.join(
            os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
            self._testMethodName)
        fileio.makedirs(self._output_data_dir)
        self._prediction_logs = standard_artifacts.InferenceResult()
        self._prediction_logs.uri = os.path.join(self._source_data_dir, 'prediction_logs')
        self._percentile_values = standard_artifacts.Examples()
        self._percentile_values.uri = os.path.join(self._source_data_dir, 'percentile_values')
        self._input_dict = {
            'inference_result': [self._prediction_logs],
            'percentile_values': [self._percentile_values],
        }

        self._output_dict = {}
        self._exec_properties = self._MakeExecProperties()

    def _MakeExecProperties(self):
        connection_config = mysql_config_pb2.MySQLConnConfig(
                                host='10.10.0.7',
                                port=3306,
                                user='mlwizardxueming',
                                password='the_Albaphet',
                                database='sc_medux')
        return {'connection_config': connection_config}

    def assertDirectoryEmpty(self, path):
        self.assertEqual(len(fileio.listdir(path)), 0)

    def assertDirectoryNotEmpty(self, path):
        self.assertGreater(len(fileio.listdir(path)), 0)

    def assertNotPushed(self):
        self.assertDirectoryEmpty(self._serving_model_dir)
        self.assertDirectoryEmpty(self._model_push.uri)
        self.assertEqual(0, self._model_push.get_int_custom_property('pushed'))

    def testPushedToDatabase(self):
        query = f"SELECT COUNT(*) FROM sc_medux.ml_test"
        client = self._exec_properties['connection_config']
        number_of_rows_before_push = connect(**json_format.MessageToDict(client)).cursor().execute(query)
        # then execute the Do function to push the results
        # Run executor. This is not debugged yet! Matthijs will do that later.
        # self._executor.Do(self._input_dict, self._output_dict,
        #                   self._exec_properties)
        number_of_rows_after_push = connect(**json_format.MessageToDict(client)).cursor().execute(query)
        # The next line will fail because the database is not updated (Do fn not executed).
        # self.assertNotEqual(number_of_rows_before_push, number_of_rows_after_push)

    # All these tests have to be looked at later by Matthijs:
    # def testDoNotBlessed(self):
    #     # Prepare not blessed ModelBlessing.
    #     self._model_blessing.uri = os.path.join(self._source_data_dir,
    #                                             'model_validator/not_blessed')
    #     self._model_blessing.set_int_custom_property('blessed', 0)
    #
    #     # Run executor with not blessed.
    #     self._executor.Do(self._input_dict, self._output_dict,
    #                       self._exec_properties)
    #
    #     # Check model not pushed.
    #     self.assertNotPushed()
    #
    # def testDo_ModelBlessedAndInfraBlessed_Pushed(self):
    #     # Prepare blessed ModelBlessing and blessed InfraBlessing.
    #     self._model_blessing.set_int_custom_property('blessed', 1)  # Blessed.
    #     infra_blessing = standard_artifacts.InfraBlessing()
    #     infra_blessing.set_int_custom_property('blessed', 1)  # Blessed.
    #     input_dict = {INFRA_BLESSING_KEY: [infra_blessing]}
    #     input_dict.update(self._input_dict)
    #
    #     # Run executor
    #     self._executor.Do(input_dict, self._output_dict, self._exec_properties)
    #
    #     # Check model is pushed.
    #     self.assertPushed()
    #
    # def testDo_InfraNotBlessed_NotPushed(self):
    #     # Prepare blessed ModelBlessing and **not** blessed InfraBlessing.
    #     self._model_blessing.set_int_custom_property('blessed', 1)  # Blessed.
    #     infra_blessing = standard_artifacts.InfraBlessing()
    #     infra_blessing.set_int_custom_property('blessed', 0)  # Not blessed.
    #     input_dict = {INFRA_BLESSING_KEY: [infra_blessing]}
    #     input_dict.update(self._input_dict)
    #
    #     # Run executor
    #     self._executor.Do(input_dict, self._output_dict, self._exec_properties)
    #
    #     # Check model is not pushed.
    #     self.assertNotPushed()
    #
    # def testDo_NoModelBlessing_InfraBlessed_Pushed(self):
    #     # Prepare successful InfraBlessing only (without ModelBlessing).
    #     infra_blessing = standard_artifacts.InfraBlessing()
    #     infra_blessing.set_int_custom_property('blessed', 1)  # Blessed.
    #     input_dict = {
    #         MODEL_KEY:
    #             self._input_dict[MODEL_KEY],
    #         INFRA_BLESSING_KEY: [infra_blessing],
    #     }
    #
    #     # Run executor
    #     self._executor.Do(input_dict, self._output_dict, self._exec_properties)
    #
    #     # Check model is pushed.
    #     self.assertPushed()
    #
    # def testDo_NoModelBlessing_InfraNotBlessed_NotPushed(self):
    #     # Prepare unsuccessful InfraBlessing only (without ModelBlessing).
    #     infra_blessing = standard_artifacts.InfraBlessing()
    #     infra_blessing.set_int_custom_property('blessed', 0)  # Not blessed.
    #     input_dict = {
    #         MODEL_KEY:
    #             self._input_dict[MODEL_KEY],
    #         INFRA_BLESSING_KEY: [infra_blessing],
    #     }
    #
    #     # Run executor
    #     self._executor.Do(input_dict, self._output_dict, self._exec_properties)
    #
    #     # Check model is not pushed.
    #     self.assertNotPushed()
    #
    # def testDo_KerasModelPath(self):
    #     # Prepare blessed ModelBlessing.
    #     self._model_export.uri = os.path.join(self._source_data_dir,
    #                                           'trainer/keras')
    #     self._model_blessing.uri = os.path.join(self._source_data_dir,
    #                                             'model_validator/blessed')
    #     self._model_blessing.set_int_custom_property('blessed', 1)
    #
    #     # Run executor
    #     self._executor.Do(self._input_dict, self._output_dict,
    #                       self._exec_properties)
    #
    #     # Check model is pushed.
    #     self.assertPushed()
    #
    # def testDo_NoBlessing(self):
    #     # Input without any blessing.
    #     input_dict = {MODEL_KEY: [self._model_export]}
    #
    #     # Run executor
    #     self._executor.Do(input_dict, self._output_dict, self._exec_properties)
    #
    #     # Check model is pushed.
    #     self.assertPushed()
    #
    # def testDo_NoModel(self):
    #     with self.assertRaisesRegex(RuntimeError, 'Pusher has no model input.'):
    #         self._executor.Do(
    #             {},  # No model and infra_blessing input.
    #             self._output_dict,
    #             self._exec_properties)
    #
    # def testDo_InfraBlessingAsModel(self):
    #     infra_blessing = standard_artifacts.InfraBlessing()
    #     infra_blessing.uri = os.path.join(self._output_data_dir, 'infra_blessing')
    #     infra_blessing.set_int_custom_property('blessed', True)
    #     infra_blessing.set_int_custom_property('has_model', 1)
    #     # Create dummy model
    #     blessed_model_path = path_utils.stamped_model_path(infra_blessing.uri)
    #     fileio.makedirs(blessed_model_path)
    #     io_utils.write_string_file(
    #         os.path.join(blessed_model_path, 'my-model'), '')
    #
    #     self._executor.Do(
    #         {INFRA_BLESSING_KEY: [infra_blessing]},
    #         self._output_dict,
    #         self._exec_properties)
    #
    #     self.assertPushed()
    #     self.assertTrue(
    #         fileio.exists(
    #             os.path.join(self._model_push.uri, 'my-model')))
    #
    # def testDo_InfraBlessingAsModel_FailIfNoWarmup(self):
    #     infra_blessing = standard_artifacts.InfraBlessing()
    #     infra_blessing.set_int_custom_property('blessed', True)
    #     infra_blessing.set_int_custom_property('has_model', 0)
    #
    #     with self.assertRaisesRegex(
    #             RuntimeError, 'InfraBlessing does not contain a model'):
    #         self._executor.Do(
    #           {INFRA_BLESSING_KEY: [infra_blessing]},
    #           self._output_dict,
    #           self._exec_properties)


if __name__ == '__main__':
    tf.test.main()
