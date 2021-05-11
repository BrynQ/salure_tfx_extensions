"""Salure TFX LC Embedding Component"""

from tfx import types
from tfx.types import standard_artifacts, channel_utils
from tfx.dsl.components.base import base_component, executor_spec
from typing import Optional, Text
from salure_tfx_extensions.components.lc_embedding import executor
from salure_tfx_extensions.types.component_specs import LCEmbeddingSpec

class LCEmbedding(base_component.BaseComponent):
    """
    Embedding for the LC with a mapping file
    """

    SPEC_CLASS = LCEmbeddingSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(self,
                 input_data: types.Channel = None,
                 mapping_file_path: Optional[Text] = None,
                 output_data: types.Channel = None,
                 name: Optional[Text] = None):

        if not output_data:
            examples_artifact = standard_artifacts.Examples()
            examples_artifact.split_names = input_data.get()[0].split_names
            output_data = channel_utils.as_channel([examples_artifact])

        spec = LCEmbeddingSpec(input_data=input_data,
                                         mapping_file_path = mapping_file_path,
                                         output_data=output_data,
                                         name=name)

        super(LCEmbedding, self).__init__(spec=spec)