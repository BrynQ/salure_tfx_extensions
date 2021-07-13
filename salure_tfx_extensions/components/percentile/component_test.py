import tensorflow as tf
from salure_tfx_extensions.components.percentile.component import PercentileComponent
from tfx.types import channel_utils
from tfx.types import standard_artifacts


class ComponentTest(tf.test.TestCase):

    def setUp(self):
        super(ComponentTest, self).setUp()
        self.inference_result = channel_utils.as_channel([standard_artifacts.InferenceResult()])

    def testConstructPercentiles(self):
        percentile = PercentileComponent(
            inference_result=self.inference_result,
            num_quantiles='20',
            instance_name='percentile')
        self.assertEqual(
            'Examples', percentile.outputs['percentile_values'].type_name)
        self.assertNotIn('inference_result', percentile.outputs.keys())


if __name__ == '__main__':
    tf.test.main()
