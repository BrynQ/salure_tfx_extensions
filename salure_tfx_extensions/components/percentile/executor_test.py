import os
import unittest
import tensorflow as tf
from salure_tfx_extensions.components.percentile import executor
from tfx.dsl.io import fileio
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs

from google.protobuf import text_format
from tensorflow_serving.apis import prediction_log_pb2


@unittest.skipIf(tf.__version__ < '2',
                 'This test uses testdata only compatible with TF 2.x')
class ExecutorTest(tf.test.TestCase):

    def setUp(self):
        super(ExecutorTest, self).setUp()
        self._source_data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'testdata')
        self._output_data_dir = os.path.join(
            os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
            self._testMethodName)
        self.component_id = 'test_component'

        # Create input dict.
        self._prediction_log = standard_artifacts.InferenceResult()
        self._prediction_log_dir = os.path.join(self._source_data_dir, 'prediction_logs')
        self._prediction_log.uri = self._prediction_log_dir
        self._PREDICTION_LOGS_FILE_NAME = "prediction_logs"

        self._input_dict = {
            standard_component_specs.INFERENCE_RESULT_KEY: [self._prediction_log],
        }

        # Create output dict.
        self.percentile_values = standard_artifacts.Examples()
        self.percentile_values_dir = os.path.join(self._output_data_dir, 'percentile_values')
        self.percentile_values.uri = self.percentile_values_dir
        self._output_dict = {
            'percentile_values': [self.percentile_values]
        }

        # Create exe properties.
        self._exec_properties = {
            'component_id': self.component_id,
            'num_quantiles': '20'
        }

        # Create context
        self._tmp_dir = os.path.join(self._output_data_dir, '.temp')
        self._context = executor.Executor.Context(
            tmp_dir=self._tmp_dir, unique_id='2')

    def _get_results(self, path, file_name, proto_type):
        results = []
        filepattern = os.path.join(path, file_name) + '-?????-of-?????.gz'
        for f in fileio.glob(filepattern):
            record_iterator = tf.compat.v1.python_io.tf_record_iterator(
                path=f,
                options=tf.compat.v1.python_io.TFRecordOptions(
                    tf.compat.v1.python_io.TFRecordCompressionType.GZIP))
            for record_string in record_iterator:
                prediction_log = proto_type()
                prediction_log.MergeFromString(record_string)
                results.append(prediction_log)
        return results

    def _verify_example_split(self, split_name):
        self.assertTrue(
            fileio.exists(
                os.path.join(self._output_examples_dir, f'Split-{split_name}')))
        results = self._get_results(
            os.path.join(self._output_examples_dir, f'Split-{split_name}'),
            executor._EXAMPLES_FILE_NAME, tf.train.Example)
        self.assertTrue(results)
        self.assertIn('classify_label', results[0].features.feature)
        self.assertIn('classify_score', results[0].features.feature)

    def testDoWithInferenceResult(self):
        # Run executor.
        percentile = executor.Executor(self._context)
        percentile.Do(self._input_dict, self._output_dict, self._exec_properties)

        # Check outputs.
        self.assertTrue(fileio.exists(self._prediction_log_dir))
        results = self._get_results(self._prediction_log_dir,
                                    self._PREDICTION_LOGS_FILE_NAME,
                                    prediction_log_pb2.PredictionLog)
        self.assertTrue(results)
        # My adaptation of the test:
        self.assertEqual(len(results[0].predict_log.response.outputs['output'].float_val), 1)
        self.assertEqual(len(results[0].predict_log.request.inputs['examples'].string_val), 1)  #TODO: test the input example contents


if __name__ == '__main__':
    tf.test.main()
