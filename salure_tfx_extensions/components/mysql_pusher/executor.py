"""MySQLPusher executor, will push the provided inference result to the database"""

from collections import OrderedDict
import os
import absl
import apache_beam as beam
import tensorflow as tf
import pymysql
from typing import Any, Dict, List, Text, Optional
from tfx import types
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from salure_tfx_extensions.proto import mysql_config_pb2
from tensorflow_serving.apis import prediction_log_pb2
from tfx.utils import io_utils
from google.protobuf import json_format

_TELEMETRY_DESCRIPTORS = ['MySQLPusher']
CUSTOM_EXPORT_FN = 'custom_export_fn'
_MODULE_FILE_KEY = 'module_file'
_PREDICTION_LOGS_FILE_NAME = 'prediction_logs'


class Executor(base_executor.BaseExecutor):
    """Executor for the BaseComponent boilerplate.
    Will read in Examples, convert them rows, and then back writing them to file as examples
    """

    def Do(self, input_dict: Dict[Text, List[types.Artifact]],
           output_dict: Dict[Text, List[types.Artifact]],
           exec_properties: Dict[Text, Any]) -> None:
        """
        Args:
          input_dict: Input dict from input key to a list of Artifacts.
            - examples: Tensorflow Examples
          output_dict: Output dict from output key to a list of Artifacts.
            - output_examples: Tensorflow Examples
          exec_properties: A dict of execution properties.
            In this case there are no items in exec_properties, as stated by BaseComponentSpec
        Returns:
          None
        """
        self._log_startup(input_dict, output_dict, exec_properties)

        predictions = artifact_utils.get_single_instance(input_dict['inference_result'])
        predictions_path = predictions.uri
        predictions_uri = io_utils.all_files_pattern(predictions_path)
        print(f"Json format prediction results saved to {predictions_path}")

        with self._make_beam_pipeline() as pipeline:
            data = (pipeline
                    | 'ReadPredictionLogs' >> beam.io.ReadFromTFRecord(
                        predictions_uri,
                        coder=beam.coders.ProtoCoder(prediction_log_pb2.PredictionLog))
                    | 'ParsePredictionLogs' >> beam.Map(parse_predictlog))

            _ = (data
                    | 'Write To MySQL db' >> _ExampleToMySQL(exec_properties))

            _ = (data
                    | 'WritePredictionLogs' >> beam.io.WriteToText(
                        file_path_prefix=os.path.join(predictions_path, _PREDICTION_LOGS_FILE_NAME),
                        num_shards=1,
                        file_name_suffix=".json")
                    )

@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
def _ExampleToMySQL(
        pipeline: beam.Pipeline,
        exec_properties: Dict[Text, any],
        table_name: Optional[Text] = 'ml_test'):

    mysql_config = mysql_config_pb2.MySQLConnConfig()
    json_format.Parse(exec_properties['connection_config'], mysql_config)

    return (pipeline
            | 'WriteMySQLDoFN' >> beam.ParDo(_WriteMySQLDoFn(mysql_config, table_name)))


class _WriteMySQLDoFn(beam.DoFn):
    """Inspired by:
    https://github.com/esakik/beam-mysql-connector/blob/master/beam_mysql/connector/io.py"""

    def __init__(
            self,
            mysql_config: mysql_config_pb2.MySQLConnConfig,
            table_name
    ):
        super(_WriteMySQLDoFn, self).__init__()

        self.mysql_config = json_format.MessageToDict(mysql_config)
        self.table_name = table_name

    def start_bundle(self):
        self._column_str = []
        self._values = []

    def process(self, element, *args, **kwargs):
        columns = []
        values = []

        for column, value in element.items():
            if column in ['periode', 'temp_contract', 'new_rooster', 'cao_code', 'hoofddienstverband', 'medewerker_id',
                          'fte_wo', 'part_time_contract', 'fte_vr', 'fte_di', 'looncomponent_extern_nummer', 'fte',
                          'type_contract', 'dagen_per_week', 'fte_do', 'fte_ma', 'werkgever_id', 'type_medewerker',
                          'boekjaar', 'expired_rooster', 'bedrag', 'trainee_time_contract', 'uren_per_week',
                          'full_time_contract', 'score']:
            #TODO: Change this when new sql table is created for the inference results
                columns.append(column)
                values.append(value)

        value_str = ", ".join(
            [
                f"{'NULL' if value is None else value}" if isinstance(value, (type(None), int, float)) else f"'{value}'"
                for value in values
            ]
        )
        self._values.append("(" + value_str + ")")

        self._column_str = "(" + ", ".join(columns) + ")"

    def finish_bundle(self):
        if len(self._values):
            client = pymysql.connect(**self.mysql_config)
            cursor = client.cursor()

            value_str = ", ".join(self._values)
            query = f"INSERT INTO {self.mysql_config['database']}.{self.table_name} {self._column_str} VALUES {value_str};"

            cursor.execute(query)
            self._values.clear()
            self._column_str = ""
            client.commit()
            cursor.close()
            client.close()


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

    example = pb.predict_log.request.inputs["examples"].string_val[0]
    example = tf.train.Example.FromString(example)

    results = parse_pb(example)
    results['score'] = predict_val

    return OrderedDict(sorted(results.items(), key=lambda t: t[0]))

def parse_pb(pb):
    results = {}
    for f, v in pb.features.ListFields():
        for kk, vv in v.items():
            for kkk, vvv in vv.ListFields():
                if len(vvv.value) == 0:
                    results[kk] = ''
                elif type(vvv.value[0]) == bytes:
                    results[kk] = vvv.value[0].decode("utf-8")
                else:
                    results[kk] = vvv.value[0]
    return results
