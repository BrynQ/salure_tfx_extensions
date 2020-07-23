import os
import importlib.machinery, importlib.util
from types import ModuleType
import absl
import dill
import base64
from typing import Any, Dict, List, Text
import tensorflow as tf
from tfx import types
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.utils import import_utils
from tfx_bsl.tfxio import tf_example_record
import tensorflow_transform.beam as tft_beam
from salure_tfx_extensions.utils import example_parsing_utils
from salure_tfx_extensions.utils import sklearn_utils
import apache_beam as beam
from apache_beam import pvalue
# import pandas as pd
import pyarrow as pa
from sklearn.pipeline import Pipeline
from google.protobuf import json_format
# from pandas_tfrecords import to_tfrecords


EXAMPLES_KEY = 'examples'
SCHEMA_KEY = 'schema'
# MODULE_FILE_KEY = 'module_file'
PREPROCESSOR_PIPELINE_NAME_KEY = 'preprocessor_pipeline_name'
PREPROCESSOR_PICKLE_KEY = 'preprocessor_pickle'
TRANSFORMED_EXAMPLES_KEY = 'transformed_examples'
TRANSFORM_PIPELINE_KEY = 'transform_pipeline'

_TELEMETRY_DESCRIPTORS = ['SKLearnTransform']

DEFAULT_PIPELINE_NAME = 'pipeline'
PIPELINE_FILE_NAME = 'pipeline.pickle'


class Executor(base_executor.BaseExecutor):
    """Executor for the SKLearnTransform Component
    Reads in Examples, and extracts a Pipeline object from a module file.
    It fits the pipeline, and writes the fit pipeline and the transformed examples to file"""

    def Do(self,
           input_dict: Dict[Text, List[types.Artifact]],
           output_dict: Dict[Text, List[types.Artifact]],
           exec_properties: Dict[Text, Any]) -> None:
        """
        Args:
          input_dict:
            - examples: Tensorflow Examples
          exec_properties:
            - preprocessor_pickle: A pickle string of the preprocessor
            - preprocessor_pipeline_name: The name of the pipeline object in the specified module file
          output_dict:
            - transformed_examples: The transformed Tensorflow Examples
            - transform_pipeline: A trained SKLearn Pipeline
        """

        self._log_startup(input_dict, output_dict, exec_properties)

        if not (len(input_dict[EXAMPLES_KEY]) == 1):
            raise ValueError('input_dict[{}] should only contain one artifact'.format(EXAMPLES_KEY))

        examples_artifact = input_dict[EXAMPLES_KEY][0]
        examples_splits = artifact_utils.decode_split_names(examples_artifact.split_names)

        train_and_eval_split = ('train' in examples_splits and 'eval' in examples_splits)
        single_split = ('single_split' in examples_artifact.uri)

        if train_and_eval_split == single_split:
            raise ValueError('Couldn\'t determine which input split to fit the pipeline on. '
                             'Exactly one split between \'train\' and \'single_split\' should be specified.')

        train_split = 'train' if train_and_eval_split else 'single_split'

        train_uri = os.path.join(examples_artifact.uri, train_split)

        # Load in the schema
        schema_path = io_utils.get_only_uri_in_dir(
            artifact_utils.get_single_uri(input_dict[SCHEMA_KEY]))
        schema = io_utils.SchemaReader().read(schema_path)
        schema_dict = json_format.MessageToDict(schema, preserving_proto_field_name=True)

        # This way a pickle bytestring could be sent over json
        sklearn_pipeline = dill.loads(base64.decodebytes(exec_properties['preprocessor_pickle'].encode('utf-8')))

        absl.logging.info('pipeline: {}'.format(sklearn_pipeline))

        with self._make_beam_pipeline() as pipeline:
            with tft_beam.Context(
                    use_deep_copy_optimization=True):
                absl.logging.info('Loading Training Examples')
                train_input_uri = io_utils.all_files_pattern(train_uri)
                preprocessor_output_uri = artifact_utils.get_single_uri(output_dict[TRANSFORM_PIPELINE_KEY])

                absl.logging.info(input_dict)
                absl.logging.info(output_dict)
                absl.logging.info('uri: {}'.format(train_uri))
                absl.logging.info('input_uri: {}'.format(train_input_uri))
                absl.logging.info('preprocessor_output_uri: {}'.format(preprocessor_output_uri))

                training_data = pipeline | 'Read Train Data' >> sklearn_utils.ReadTFRecordToPandas(
                    file_pattern=train_input_uri,
                    schema=schema,
                    split_name='Train',  # Is just for naming the beam operations
                    telemetry_descriptors=_TELEMETRY_DESCRIPTORS
                )

                training_data | 'Logging Pandas DataFrame' >> beam.Map(
                    lambda x: absl.logging.info('dataframe: {}'.format(x)))
                training_data | 'Log DataFrame head' >> beam.Map(lambda x: print(x.head().to_string()))

                preprocessor_pcoll = pipeline | beam.Create([sklearn_pipeline])

                def fit_sklearn_preprocessor(df, sklearn_preprocessor_pipeline):
                    sklearn_preprocessor_pipeline.fit(df)
                    yield pvalue.TaggedOutput('fit_preprocessor', sklearn_preprocessor_pipeline)
                    yield pvalue.TaggedOutput('transformed_df', sklearn_preprocessor_pipeline.transform(df))

                results = training_data | 'Fit Preprocessing Pipeline' >> beam.FlatMap(
                    fit_sklearn_preprocessor,
                    pvalue.AsSingleton(preprocessor_pcoll)).with_outputs()

                fit_preprocessor = results.fit_preprocessor
                transformed_df = results.transformed_df

                fit_preprocessor | 'Logging Fit Preprocessor' >> beam.Map(
                    lambda x: absl.logging.info('fit_preprocessor: {}'.format(x)))

                transformed_df | 'Logging Transformed DF head' >> beam.Map(
                    lambda x: absl.logging.info('transformed_df head: {}'.format(x)))

                fit_preprocessor | sklearn_utils.WriteSKLearnModelToFile(
                    os.path.join(preprocessor_output_uri, PIPELINE_FILE_NAME))

                transformed_df | 'Write Train Data to File' >> sklearn_utils.WriteDataFrame(
                    os.path.join(output_dict[TRANSFORMED_EXAMPLES_KEY][0].uri, train_split))

                if train_and_eval_split:
                    test_split = 'eval'
                    test_uri = os.path.join(examples_artifact.uri, test_split)
                    test_input_uri = io_utils.all_files_pattern(test_uri)

                    test_data = pipeline | 'Read Test Data' >> sklearn_utils.ReadTFRecordToPandas(
                        file_pattern=test_input_uri,
                        schema=schema,
                        split_name='Test',  # Is just for naming the beam operations
                        telemetry_descriptors=_TELEMETRY_DESCRIPTORS
                    )

                    def transform_data(df, sklearn_preprocessor_pipeline):
                        return sklearn_preprocessor_pipeline.transform(df)

                    transformed_test_data = test_data | 'Transform Test Data' >> beam.Map(
                        transform_data,
                        pvalue.AsSingleton(fit_preprocessor))

                    transformed_test_data | 'Write Test Data to File' >> sklearn_utils.WriteDataFrame(
                        os.path.join(output_dict[TRANSFORMED_EXAMPLES_KEY][0].uri, test_split))
