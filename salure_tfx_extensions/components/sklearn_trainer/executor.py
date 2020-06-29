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
    def Do(self):
        pass
