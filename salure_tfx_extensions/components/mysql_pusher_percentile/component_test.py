import tensorflow as tf
from salure_tfx_extensions.components.mysql_pusher_percentile.component import MySQLPusher
from salure_tfx_extensions.components.mysql_pusher_percentile import executor
from salure_tfx_extensions.proto import mysql_config_pb2
from tfx.types import channel_utils
from tfx.types import standard_artifacts


class ComponentTest(tf.test.TestCase):

    class _MyCustomPusherExecutor(executor.Executor):
        """Mock class to test custom executor injection."""
        pass

    def setUp(self):
        super(ComponentTest, self).setUp()
        self._prediction_log = channel_utils.as_channel([standard_artifacts.InferenceResult()])
        self._percentile_values = channel_utils.as_channel([standard_artifacts.Examples()])
        self._connection_config = mysql_config_pb2.MySQLConnConfig(
                                            host="10.10.0.7",
                                            port=3306,
                                            user="mlwizardxueming",
                                            password="the_Albaphet",
                                            database="sc_medux")
        self._table_name = "ml_results"
        self._inference_id = "inference_id"

    def testConstructPusherPercentiles(self):
        pusher = MySQLPusher(
            inference_result=self._prediction_log,
            percentile_values=self._percentile_values,
            connection_config=self._connection_config,
            table_name=self._table_name,
            inference_id=self._inference_id,
            instance_name="pusher_percentile")
        self.assertEqual(
            standard_artifacts.InferenceResult.TYPE_NAME,
            pusher.inputs['inference_result'].type_name)
        self.assertCountEqual(
                pusher.inputs.keys(),
                ['inference_result', 'percentile_values'])


if __name__ == '__main__':
    tf.test.main()
