"""A trainer component for SKLearn models"""

from typing import Optional, Text, Dict, Any, Union

from tfx import types
from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.types import standard_artifacts
from salure_tfx_extensions.types import standard_artifacts as stfxe_artifacts
from tfx.types import artifact_utils
from tfx.proto import example_gen_pb2
from tfx.components.example_gen import utils
from tfx.types import channel_utils
from salure_tfx_extensions.components.sklearn_transform import executor
from salure_tfx_extensions.types.component_specs import SKLearnTransformSpec


class SKLearnTransform(base_component.BaseComponent):
    """A component for preprocessing examples using SKLearn"""

    SPEC_CLASS = SKLearnTransformSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    # TODO
    def __init__(self,
                 examples: types.Channel,
                 module_file: Text,
                 preprocessor_pipeline_name: Optional[Text] = None,
                 instance_name: Optional[Text] = None):
                 # enable_cache: Optional[bool] = None):

        transform_pipeline = types.Channel(
            type=stfxe_artifacts.SKLearnPrepocessor, artifacts=[stfxe_artifacts.SKLearnPrepocessor()]
        )

        transformed_examples = types.Channel(
            type=standard_artifacts.Examples, artifacts=[standard_artifacts.Examples()]
        )

        spec = SKLearnTransformSpec(
            examples=examples,
            module_file=module_file,
            preprocessor_pipeline_name=preprocessor_pipeline_name,
            transform_pipeline=transform_pipeline,
            transformed_examples=transformed_examples,
        )

        super(SKLearnTransform, self).__init__(
            spec=spec,
            instance_name=instance_name,
            # enable_cache=enable_cache,
        )
