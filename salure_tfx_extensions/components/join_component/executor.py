"""Base executor, reads Examples, **writes them to logs** and back to Example"""

import os
import absl
import apache_beam as beam
import tensorflow as tf
from typing import Any, Dict, List, Text
from tfx import types
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.utils import path_utils
# from tfx_bsl.tfxio import tf_example_record
from tfx_bsl.tfxio import tf_example_record
from salure_tfx_extensions.utils import sklearn_utils

LEFT_EXAMPLES_KEY = 'left_examples'
RIGHT_EXAMPLES_KEY = 'right_examples'
OUTPUT_EXAMPLES_KEY = 'examples'
_TELEMETRY_DESCRIPTORS = ['JoinComponent']

KEY_NAME = 'key'

LEFT_SCHEMA_KEY = 'left_schema'
RIGHT_SCHEMA_KEY = 'right_schema'

# DEFAULT_FILE_NAME = 'data_tfrecord'


class Executor(base_executor.BaseExecutor):
    """Executor for the BaseComponent boilerplate.
    Will read in Examples, convert them rows, and then back writing them to file as examples
    """

    def _validate_inputs(self,
                         left_examples_key: Text,
                         right_examples_key: Text,
                         output_key: Text,
                         input_dict: Dict[Text, List[types.Artifact]],
                         output_dict: Dict[Text, List[types.Artifact]]) -> (Text, Text):
        """Returns the split uris for the left and right examples respectively"""
        if left_examples_key not in input_dict:
            raise ValueError('\'{}\' is missing from input_dict'.format(left_examples_key))

        if right_examples_key not in input_dict:
            raise ValueError('\'{}\' is missing from input_dict'.format(right_examples_key))

        output_examples_artifacts = output_dict[output_key]

        # Assumed input_dict['examples'] and output_dict['output_examples'] contain only one Artifact
        if not (1 == len(output_examples_artifacts) == len(input_dict[left_examples_key]) == len(
                input_dict[right_examples_key])):
            raise ValueError('input_dict[{}], input_dict[{}] and output_dict[{}] should have only one artifact'.format(
                left_examples_key,
                right_examples_key,
                output_key))

        # Validate left schema
        if LEFT_SCHEMA_KEY not in input_dict:
            raise ValueError('schema for left data is missing')

        if RIGHT_SCHEMA_KEY not in input_dict:
            raise ValueError('schema for right data is missing')

        # Validate left examples artifact
        left_examples_artifact = input_dict[left_examples_key][0]
        left_examples_splits = artifact_utils.decode_split_names(left_examples_artifact.split_names)

        if len(left_examples_artifact) != 1:
            raise ValueError('input_dict[{}] artifact doesn\'t have a single split'.format(left_examples_key))

        left_examples_split = left_examples_splits[0]

        # Validate right examples artifact
        right_examples_artifact = input_dict[right_examples_key][0]
        right_examples_splits = artifact_utils.decode_split_names(right_examples_artifact.split_names)

        if len(right_examples_splits) != 1:
            raise ValueError('input_dict[{}] artifact doesn\'t have a single split'.format(right_examples_key))

        right_examples_split = right_examples_splits[0]

        return left_examples_split, right_examples_split

    def _validate_exec_properties(self,
                                  key_name: Text,
                                  exec_properties: Dict[Text, Any]):
        if not isinstance(exec_properties[key_name], str):
            raise ValueError('exec_properties[{}] needs to be str'.format(key_name))

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

        left_split, right_split = self._validate_inputs(
            LEFT_EXAMPLES_KEY,
            RIGHT_EXAMPLES_KEY,
            KEY_NAME,
            input_dict,
            output_dict
        )

        self._validate_exec_properties(
            KEY_NAME,
            exec_properties)

        with self._make_beam_pipeline() as pipeline:
            absl.logging.info('Loading left examples')

            left_split_name, left_split_uri = left_split
            left_input_uri = io_utils.all_files_pattern(left_split_uri)

            # Load in the schema
            schema_path = io_utils.get_only_uri_in_dir(
                artifact_utils.get_single_uri(input_dict[LEFT_SCHEMA_KEY]))
            left_schema = io_utils.SchemaReader().read(schema_path)

            # Load in the schema
            schema_path = io_utils.get_only_uri_in_dir(
                artifact_utils.get_single_uri(input_dict[RIGHT_SCHEMA_KEY]))
            right_schema = io_utils.SchemaReader().read(schema_path)

            # left_data = (pipeline
            #                 | 'ReadLeftExamplesFromTFRecord' >> beam.io.ReadFromTFRecord(
            #                     file_pattern=left_input_uri)
            #                 | 'ParseLeftExamples' >> beam.Map(tf.train.Example.FromString))

            left_df = sklearn_utils.ReadTFRecordToPandas(
                file_pattern=left_split_uri,
                schema=left_schema,
                split_name=left_split_name,
                telemetry_descriptors=_TELEMETRY_DESCRIPTORS,
            )



            absl.logging.info('Loading right examples')

            right_split_name, right_split_uri = right_split
            right_input_uri = io_utils.all_files_pattern(right_split_uri)

            # right_data = (pipeline
            #                 | 'ReadExamplesFromTFRecord' >> beam.io.ReadFromTFRecord(
            #                     file_pattern=left_input_uri)
            #                 | 'ParseLeftExamples' >> beam.Map(tf.train.Example.FromString))

        # TODO Remove
        # return 0
        with self._make_beam_pipeline() as pipeline:
            for split, uri in split_uris:
                absl.logging.info('Loading examples for split {}'.format(split))
                input_uri = io_utils.all_files_pattern(uri)
                input_tfxio = tf_example_record.TFExampleRecord(
                    file_pattern=input_uri,
                    telemetry_descriptors=_TELEMETRY_DESCRIPTORS
                )

                absl.logging.info(input_dict)
                absl.logging.info(output_dict)
                absl.logging.info('split: {}'.format(split))
                absl.logging.info('uri: {}'.format(uri))
                absl.logging.info('input_uri: {}'.format(input_uri))

                # output_path = artifact_utils.get_split_uri(output_dict[OUTPUT_EXAMPLES_KEY],
                #                                            split)
                output_path = os.path.join(output_examples_artifacts[0].uri, split)

                # loading the data and displaying
                # data = pipeline | 'TFXIORead[{}]'.format(split) >> input_tfxio.BeamSource()
                data = (pipeline
                        | 'ReadExamplesFromTFRecord[{}]'.format(split) >> beam.io.ReadFromTFRecord(
                            file_pattern=input_uri)
                        | 'ParseExamples[{}]'.format(split) >> beam.Map(tf.train.Example.FromString))

                # logging the rows, and writing them back to file
                # this is of course not as efficient as copying the input files
                # but this is meant as a boilerplate component to work from
                data | 'Printing data from {}'.format(split) >> beam.Map(absl.logging.info)
                (data
                 | 'Serializing Examples [{}]'.format(split) >> beam.Map(
                            lambda x: x.SerializeToString(deterministic=True))
                 | 'WriteSplit[{}]'.format(split) >> _WriteSplit(output_path))


@beam.ptransform_fn
@beam.typehints.with_input_types(bytes)
@beam.typehints.with_output_types(beam.pvalue.PDone)
def _WriteSplit(example_split: beam.pvalue.PCollection,
                output_split_path: Text) -> beam.pvalue.PDone:
    """Shuffles and writes output split."""
    return (example_split
            | 'Shuffle' >> beam.transforms.Reshuffle()
            | 'Write' >> beam.io.WriteToTFRecord(
                os.path.join(output_split_path, DEFAULT_FILE_NAME),
                file_name_suffix='.gz'))
