"""A trainer component for SKLearn models"""

import os
import pickle
import absl
import apache_beam as beam
import tensorflow as tf
from typing import Any, Dict, List, Text, Tuple
from tfx import types
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.utils import path_utils
from salure_tfx_extensions.utils import example_parsing_utils
# from tfx_bsl.tfxio import tf_example_record

EXAMPLES_KEY = 'examples'
MODEL_KEY = 'model'
_TELEMETRY_DESCRIPTORS = ['SKLearnTrainer']


def _get_train_and_eval_uris(artifact: types.Artifact, splits: List[Text]) -> Tuple[Text, Text]:
    if not ('train' in splits and 'eval' in splits):
        raise ValueError('Missing \'train\' and \'eval\' splits in \'examples\' artifact,'
                         'got {} instead'.format(splits))
    return (os.path.join(artifact.uri, 'train'),
            os.path.join(artifact.uri, 'eval'))


class Executor(base_executor.BaseExecutor):
    """Executor for the SKLearnTrainer Component

    Takes in Examples, parses them, and trains an SKLearnModel on it
    """

    # TODO
    def Do(self, input_dict: Dict[Text, List[types.Artifact]],
           output_dict: Dict[Text, List[types.Artifact]],
           exec_properties: Dict[Text, Any]) -> None:
        """
        Args:
          input_dict:
            - examples: Examples used for training, must include 'train' and 'eval' splits
          output_dict:
            - model: The trained SKLearnModel
          exec_properties:
            - model_pickle: A bytestring contain a pickled SKLearn model

        """
        self._log_startup(input_dict, output_dict, exec_properties)

        if 'examples' not in input_dict:
            raise ValueError('\'examples\' is missing in input dict')
        if 'model_pickle' not in exec_properties:
            raise ValueError('\'model_pickle\' is missing in exec_properties')

        model = pickle.loads(exec_properties['model_pickle'])

        absl.logging.info(model)

        split_uris = []

        if not len(input_dict[EXAMPLES_KEY]) == 1:
            raise ValueError('input_dict[{}] should contain only 1 artifact'.format(
                EXAMPLES_KEY))

        artifact = input_dict[EXAMPLES_KEY][0]
        splits = artifact_utils.decode_split_names(artifact.split_names)

        train_uri, eval_uri = _get_train_and_eval_uris(artifact, splits)

        with self._make_beam_pipeline() as pipeline:
            training_data = (pipeline
                             | 'ReadTrainingExamplesFromTFRecord' >> beam.io.ReadFromTFRecord(
                                file_pattern=train_uri)
                             | 'ParseTrianingExamples' >> beam.Map(tf.train.Example.FromString))

            # training_data_rows is PCollection of List[Any]
            training_data_rows = training_data | 'Training Example to rows' >> beam.Map(
                example_parsing_utils.example_to_list)

            # TODO: Support label key in the tf.examples
            # TODO: Make it possible to train unsupervised models
            # TODO: Use CombineFn to aggregate tf.examples into single 2D list
                # tip: parse tf.Example to python list using stfxe.utils.example_parsing





