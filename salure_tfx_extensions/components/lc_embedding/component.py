"""Custom LC Embedding Component"""

from tfx import types
from tfx.types import standard_artifacts, channel_utils
from tfx.dsl.components.base import base_component, executor_spec
from typing import Optional, Text
from salure_tfx_extensions.components.lc_embedding import executor
from salure_tfx_extensions.types.component_specs import LCEmbeddingSpec
from tfx.utils import json_utils

class LCEmbedding(base_component.BaseComponent):
    """
    Embedding the input data with an LC mapping file in csv format
    """

    SPEC_CLASS = LCEmbeddingSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(self,
                 input_data: types.Channel = None,
                 mapping_file_path: Optional[Text] = None,
                 output_data: types.Channel = None,
                 feature_description: Optional[Text] = None,
                 instance_name: Optional[Text] = None):

        if not output_data:
            examples_artifact = standard_artifacts.Examples()
            output_data = channel_utils.as_channel([examples_artifact])

        spec = LCEmbeddingSpec(input_data=input_data,
                                         feature_description=json_utils.dumps(feature_description),
                                         mapping_file_path=mapping_file_path, output_data=output_data)

        super(LCEmbedding, self).__init__(spec=spec, instance_name=instance_name)