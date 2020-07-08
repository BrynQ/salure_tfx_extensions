import os
import absl
import apache_beam
import tensorflow
from typing import Any, Dict, List, Text
from tfx import types
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx_bsl.tfxio import tf_example_record
import apache_beam as beam


EXAMPLES_KEY = 'examples'
MODULE_FILE_KEY = 'module_file'
PREPROCESSOR_PIPELINE_NAME_KEY = 'preprocessor_pipeline_name'
TRANSFORMED_EXAMPLES_KEY = 'transformed_examples'
TRANSFORM_PIPELINE_KEY = 'transform_pipeline'

_TELEMETRY_DESCRIPTORS = ['SKLearnTransform']


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
            - module_file: String file path to a module file
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
        absl.logging.info('train_uri: {}'.format(train_uri))

        with self._make_beam_pipeline() as pipeline:
            absl.logging.info('Loading Training Examples')
            train_input_uri = io_utils.all_files_pattern(train_uri)

            input_tfxio = tf_example_record.TFExampleRecord(
                file_pattern=train_input_uri,
                telemetry_descriptors=_TELEMETRY_DESCRIPTORS
            )

            absl.logging.info(input_dict)
            absl.logging.info(output_dict)
            absl.logging.info('uri: {}'.format(train_uri))
            absl.logging.info('input_uri: {}'.format(train_input_uri))

            data = pipeline | 'TFXIORead Train Files' >> input_tfxio.BeamSource()
            data | 'Printing data from Train Files' >> beam.Map(absl.logging.info)



