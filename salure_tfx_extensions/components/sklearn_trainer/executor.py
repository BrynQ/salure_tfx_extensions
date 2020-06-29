"""A trainer component for SKLearn models"""

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

EXAMPLES_KEY = 'examples'
MODEL_KEY = 'model'
_TELEMETRY_DESCRIPTOS = ['SKLearnTrainer']


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
            - model: The SKLearnModel
          exec_properties:
            - module_pickle: A bytestring contain a pickled SKLearn model

        """
        self._log_startup(input_dict, output_dict, exec_properties)

        if 'examples' not in input_dict:
            raise ValueError('\'examples\' is missing in input dict')
        if 'module_file' not in exec_properties:
            raise ValueError('\'module_file\' is missing in exec_properties')


