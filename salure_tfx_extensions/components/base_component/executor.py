# TODO: fill in, to which base type it gets converted
"""Base executor, reads Examples, converts them to <fill in> and back to Example"""

import os
import apache_beam as beam
import tensorflow as tf
from typing import Any, Dict, List, Text
from tfx import types
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.utils import path_utils

# Path to store model eval results for validation.
CURRENT_MODEL_EVAL_RESULT_PATH = 'eval_results/current_model/'
BLESSED_MODEL_EVAL_RESULT_PATH = 'eval_results/blessed_model/'


class Executor(base_executor.BaseExecutor):
    """Executor for the BaseComponent boilerplate.
    Will read in Examples, convert them rows, and then back writing them to file as examples"""

    def Do(self, input_dict: Dict[Text, List[types.Artifact]],
           output_dict: Dict[Text, List[types.Artifact]],
           exec_properties: Dict[Text, Any]) -> None:
        """Validate current model against last blessed model.
    Args:
      input_dict: Input dict from input key to a list of Artifacts.
        - examples: Tensorflow Examples
      output_dict: Output dict from output key to a list of Artifacts.
        - examples: Tensorflow Examples
      exec_properties: A dict of execution properties.
        In this case there are no items in exec_properties, as stated by BaseComponentSpec
    Returns:
      None
    """
        self._log_startup(input_dict, output_dict, exec_properties)

