"""MySQLPusher executor to calculate the threshold"""

from tensorflow_serving.apis import prediction_log_pb2
from tfx.components.base import base_executor
import apache_beam as beam
from tfx.types import artifact_utils
from tfx.utils import io_utils


def parse_predictlog(pb):
    predict_val = None

    response_tensor = pb.predict_log.response.outputs["output"]
    if len(response_tensor.half_val) != 0:
        predict_val = response_tensor.half_val[0]
    elif len(response_tensor.float_val) != 0:
        predict_val = response_tensor.float_val[0]
    elif len(response_tensor.double_val) != 0:
        predict_val = response_tensor.double_val[0]
    elif len(response_tensor.int_val) != 0:
        predict_val = response_tensor.int_val[0]
    elif len(response_tensor.string_val) != 0:
        predict_val = response_tensor.string_val[0]
    elif len(response_tensor.int64_val) != 0:
        predict_val = response_tensor.int64_val[0]
    elif len(response_tensor.bool_val) != 0:
        predict_val = response_tensor.bool_val[0]
    elif len(response_tensor.uint32_val) != 0:
        predict_val = response_tensor.uint32_val[0]
    elif len(response_tensor.uint64_val) != 0:
        predict_val = response_tensor.uint64_val[0]

    if predict_val is None:
        ValueError("Encountered response tensor with unknown value")

    return predict_val

class Executor(base_executor.BaseExecutor):

    def Do(self, input_dict, output_dict, exec_properties):
        self._log_startup(input_dict, output_dict, exec_properties)
        predictions = artifact_utils.get_single_instance(input_dict['inference_result'])
        predictions_path = predictions.uri
        predictions_uri = io_utils.all_files_pattern(predictions_path)

        num_quantiles = int(exec_properties['num_quantiles']) + 1

        output_examples_uri = artifact_utils.get_single_uri(output_dict['percentile_values'])

        with beam.Pipeline() as pipeline:
            train_data = (pipeline
                          | 'ReadPredictionLogs' >> beam.io.ReadFromTFRecord(
                        predictions_uri,
                        coder=beam.coders.ProtoCoder(prediction_log_pb2.PredictionLog))
                          | 'ParsePredictionLogs' >> beam.Map(parse_predictlog))

            quantiles = (train_data | 'Quantiles globally' >> beam.transforms.stats.ApproximateQuantiles.Globally(
                num_quantiles=num_quantiles)
                         | 'WriteToFile' >> beam.io.WriteToText(
                        file_path_prefix=output_examples_uri + '/percentile_values',
                        shard_name_template='',
                        file_name_suffix='.txt'))