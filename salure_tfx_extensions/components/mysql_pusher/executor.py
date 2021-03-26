"""MySQLPusher executor, will push the provided inference result to the database"""

import os
import absl
import apache_beam as beam
import tensorflow as tf
import pymysql
from typing import Any, Dict, List, Text, Optional
from tfx import types
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.proto import example_gen_pb2
from salure_tfx_extensions.proto import mysql_config_pb2
from tensorflow_serving.apis import prediction_log_pb2
from tfx.components.util import udf_utils
from tfx.utils import io_utils
from tfx.utils import path_utils
# from tfx_bsl.tfxio import tf_example_record
from tfx_bsl.tfxio import tf_example_record
# from tfx.components.bulk_inferrer.executor import _PREDICTION_LOGS_FILE_NAME
from tfx.utils import import_utils
from google.protobuf.message import Message
from google.protobuf import json_format
from google.protobuf.descriptor import FieldDescriptor

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
        # predictions_path = os.path.join(predictions.uri, _PREDICTION_LOGS_FILE_NAME)
        predictions_path = predictions.uri
        predictions_uri = io_utils.all_files_pattern(predictions_path)
        print(f"Json format prediction results saved to {predictions_path}")
        # custom_fn = udf_utils.get_fn(exec_properties, 'custom_export_fn')

        # custom_fn = import_utils.import_func_from_source(exec_properties[_MODULE_FILE_KEY], CUSTOM_EXPORT_FN)

        # if EXAMPLES_KEY not in input_dict:
        #     raise ValueError('\'{}\' is missing from input_dict'.format(EXAMPLES_KEY))
        #
        # split_uris = []
        # output_examples_artifacts = output_dict[OUTPUT_EXAMPLES_KEY]
        #
        # # Assumed input_dict['examples'] and output_dict['output_examples'] contain only one Artifact
        # if not (1 == len(output_examples_artifacts) == len(input_dict[EXAMPLES_KEY])):
        #     raise ValueError('input_dict[{}] and output_dict[{}] should have length 1'.format(
        #         EXAMPLES_KEY,
        #         OUTPUT_EXAMPLES_KEY))
        #
        # for artifact in input_dict[EXAMPLES_KEY]:
        #     for split in artifact_utils.decode_split_names(artifact.split_names):
        #         uri = os.path.join(artifact.uri, split)
        #         split_uris.append((split, uri))

        with self._make_beam_pipeline() as pipeline:
            data = (pipeline
                    | 'ReadPredictionLogs' >> beam.io.ReadFromTFRecord(
                        predictions_uri,
                        coder=beam.coders.ProtoCoder(prediction_log_pb2.PredictionLog))
                    | 'ParsePredictionLogs' >> beam.Map(parse_predictlog))

            # _ = (data
            #         | 'Log Parsing results' >> beam.Map(absl.logging.info))


            _ = (data
                    | 'Write To MySQL db' >> _ExampleToMySQL(exec_properties))

            _ = (data
                    | 'WritePredictionLogs' >> beam.io.WriteToText(
                        file_path_prefix=os.path.join(predictions_path, _PREDICTION_LOGS_FILE_NAME),
                        num_shards=1,
                        file_name_suffix=".json")
                    )
                    # | 'write file' >> beam.io.WriteToText(files_output))
                    # | 'Log PredictionLogs' >> beam.Map(absl.logging.info))
                    # | 'ParsePredictionLogs' >> beam.Map(protobuf_to_dict))

            # _ = (data
            #      | 'Log PredictionLogs' >> beam.Map(absl.logging.info))



            # for split, uri in split_uris:
            #     absl.logging.info('Loading examples for split {}'.format(split))
            #     input_uri = io_utils.all_files_pattern(uri)
            #     input_tfxio = tf_example_record.TFExampleRecord(
            #         file_pattern=input_uri,
            #         telemetry_descriptors=_TELEMETRY_DESCRIPTORS
            #     )
            #
            #     absl.logging.info(input_dict)
            #     absl.logging.info(output_dict)
            #     absl.logging.info('split: {}'.format(split))
            #     absl.logging.info('uri: {}'.format(uri))
            #     absl.logging.info('input_uri: {}'.format(input_uri))
            #
            #     # output_path = artifact_utils.get_split_uri(output_dict[OUTPUT_EXAMPLES_KEY],
            #     #                                            split)
            #     output_path = os.path.join(output_examples_artifacts[0].uri, split)
            #
            #     # loading the data and displaying
            #     # data = pipeline | 'TFXIORead[{}]'.format(split) >> input_tfxio.BeamSource()
            #     data = (pipeline
            #             | 'ReadExamplesFromTFRecord[{}]'.format(split) >> beam.io.ReadFromTFRecord(
            #                 file_pattern=input_uri)
            #             | 'ParseExamples[{}]'.format(split) >> beam.Map(tf.train.Example.FromString))
            #
            #     # logging the rows, and writing them back to file
            #     # this is of course not as efficient as copying the input files
            #     # but this is meant as a boilerplate component to work from
            #     data | 'Printing data from {}'.format(split) >> beam.Map(absl.logging.info)
            #     (data
            #      | 'Serializing Examples [{}]'.format(split) >> beam.Map(
            #                 lambda x: x.SerializeToString(deterministic=True))
            #      | 'WriteSplit[{}]'.format(split) >> _WriteSplit(output_path))


# @beam.ptransform_fn
# @beam.typehints.with_input_types(bytes)
# @beam.typehints.with_output_types(beam.pvalue.PDone)
# def _WriteSplit(example_split: beam.pvalue.PCollection,
#                 output_split_path: Text) -> beam.pvalue.PDone:
#     """Shuffles and writes output split."""
#     return (example_split
#             | 'Shuffle' >> beam.transforms.Reshuffle()
#             | 'Write' >> beam.io.WriteToTFRecord(
#                 os.path.join(output_split_path, DEFAULT_FILE_NAME),
#                 file_name_suffix='.gz'))

@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
def _ExampleToMySQL(
        pipeline: beam.Pipeline,
        exec_properties: Dict[Text, any],
        table_name: Optional[Text] = 'ml_test'):
    # conn_config = example_gen_pb2.CustomConfig()
    # json_format.Parse(exec_properties['custom_config'], conn_config)

    mysql_config = mysql_config_pb2.MySQLConnConfig()
    json_format.Parse(exec_properties['connection_config'], mysql_config)
    # conn_config.custom_config.Unpack(mysql_config)

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
        self._queries = []

    def process(self, element, *args, **kwargs):
        columns = []
        values = []

        for column, value in element.items():
            columns.append(column)
            values.append(value)

        column_str = ", ".join(columns)
        value_str = ", ".join(
            [
                f"{'NULL' if value is None else value}" if isinstance(value, (type(None), int, float)) else f"'{value}'"
                for value in values
            ]
        )

        query = f"INSERT INTO {self.mysql_config['database']}.{self.table_name} ({column_str}) VALUES({value_str});"

        self._queries.append(query)

    def finish_bundle(self):
        if len(self._queries):
            client = pymysql.connect(**self.mysql_config)
            cursor = client.cursor()

            final_query = "\n".join(self._queries)

            absl.logging.info(final_query)

            cursor.execute(final_query)
            self._queries.clear()

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
    return results


# protobuf_to_dict is from https://github.com/benhodgson/protobuf-to-dict

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


TYPE_CALLABLE_MAP = {
    FieldDescriptor.TYPE_DOUBLE: float,
    FieldDescriptor.TYPE_FLOAT: float,
    FieldDescriptor.TYPE_INT32: int,
    FieldDescriptor.TYPE_INT64: int,
    FieldDescriptor.TYPE_UINT32: int,
    FieldDescriptor.TYPE_UINT64: int,
    FieldDescriptor.TYPE_SINT32: int,
    FieldDescriptor.TYPE_SINT64: int,
    FieldDescriptor.TYPE_FIXED32: int,
    FieldDescriptor.TYPE_FIXED64: int,
    FieldDescriptor.TYPE_SFIXED32: int,
    FieldDescriptor.TYPE_SFIXED64: int,
    FieldDescriptor.TYPE_BOOL: bool,
    FieldDescriptor.TYPE_STRING: str,
    FieldDescriptor.TYPE_BYTES: lambda b: b.encode("base64"),
    FieldDescriptor.TYPE_ENUM: int,
}

def dump_object(pb):
    print ('---- Dump object ----')
    for descriptor in pb.DESCRIPTOR.fields:
        print (f"descripter: {descriptor}")
        value = getattr(pb, descriptor.name)
        print(f"descripter value: {value}")
        print(f"descripter type: {descriptor.type}")
        print(f"descripter label: {descriptor.label}")

        if descriptor.type == descriptor.TYPE_MESSAGE:
            if descriptor.label == descriptor.LABEL_REPEATED:
                map(dump_object, value)
            else:
                dump_object(value)
        elif descriptor.type == descriptor.TYPE_ENUM:
            enum_name = descriptor.enum_type.values[value].name
            print (descriptor.full_name, enum_name)
        else:
            print (descriptor.full_name, value)

def protobuf_to_dict(pb, type_callable_map=TYPE_CALLABLE_MAP, use_enum_labels=False):
    result_dict = {}
    extensions = {}
    print ("------bf to dict -----------------")
    absl.logging.info(pb)
    print ("=====================================")
    for field, value in pb.ListFields():
        try:
            print(f"\n=***=loop in=***=")
            print(f"Field: {field}")
            print(f"Field name: {field.name}")
            print(f"Field type: {field.type}")
            print(f"Field label: {field.label}")
            print(f"Field is_extension: {field.is_extension}")
            print(f"Field number: {field.number}")
            print(f"Field enum_type: {field.enum_type}")
            print(f"value: {value}")
        except:
            print ("aya")
        type_callable = _get_field_value_adaptor(pb, field, type_callable_map, use_enum_labels)
        if field.label == FieldDescriptor.LABEL_REPEATED:  # to see if the field is repeated
            type_callable = repeated(type_callable)

        if field.is_extension:
            extensions[str(field.number)] = type_callable(value)
            extensions[str(field.number)] = type_callable(value)
            continue

        result_dict[field.name] = type_callable(value)

    if extensions:
        result_dict[EXTENSION_CONTAINER] = extensions
    return result_dict

def _get_field_value_adaptor(pb, field, type_callable_map=TYPE_CALLABLE_MAP, use_enum_labels=False):
    print("===========_get_field_value_adaptor=======================")
    print(f"field.name: {field.name}")
    print(f"field.type: {field.type}")
    print(f"field.label: {field.label}")
    print(f"field.number: {field.number}")
    absl.logging.info(pb)
    print("===========_finish_get_field_value_adaptor=======================")

    # if field.type == FieldDescriptor.TYPE_MESSAGE:
    if field.name == "features":
        # recursively encode protobuf sub-message
        print (f"----return 1----")
        return lambda pb: protobuf_to_dict(pb,
            type_callable_map=type_callable_map,
            use_enum_labels=use_enum_labels)

    # if use_enum_labels:
    # # if use_enum_labels and field.type == FieldDescriptor.TYPE_ENUM:
    #     print(f"----return 2----")
    #     return lambda value: enum_label_name(field, value)

    if field.type in type_callable_map:
        print(f"----return 3----")
        return type_callable_map[field.type]

    raise TypeError("Field %s.%s has unrecognised type id %d" % (
        pb.__class__.__name__, field.name, field.type))

def repeated(type_callable):
    return lambda value_list: [type_callable(value) for value in value_list]

def enum_label_name(field, value):
    return field.enum_type.values_by_number[int(value)].name



