"""Salure TFX Uploaded_files Embedding Component"""

from tfx import types
from tfx.types import standard_artifacts, channel_utils
from tfx.dsl.components.base import base_component, executor_spec
from typing import Optional, Text
from salure_tfx_extensions.components.uploaded_files_embedding import executor
from salure_tfx_extensions.types.component_specs import UploadedfilesEmbeddingSpec
from tfx.utils import json_utils


class UploadedfilesEmbeddingComponent(base_component.BaseComponent):
    SPEC_CLASS = UploadedfilesEmbeddingSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(self, input_data: types.Channel = None,
                 mapping_data: types.Channel = None,
                 output_data: types.Channel = None,
                 feature_description: Optional[Text] = None,
                 instance_name: Optional[Text] = None):
        if not output_data:
            examples_artifact = standard_artifacts.Examples()
            output_data = channel_utils.as_channel([examples_artifact])

        spec = UploadedfilesEmbeddingSpec(input_data=input_data, mapping_data=mapping_data,
                                                   output_data=output_data,
                                                   feature_description=json_utils.dumps(feature_description))
        super(UploadedfilesEmbeddingComponent, self).__init__(spec=spec, instance_name=instance_name)
