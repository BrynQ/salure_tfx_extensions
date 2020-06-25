from typing import Optional, Text, Dict, Any, Union

from tfx import types
from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.types import standard_artifacts
from tfx.types import artifact_utils
from tfx.proto import example_gen_pb2
from tfx.components.example_gen import utils
from tfx.types import channel_utils
from salure_tfx_extensions.components.base_component import executor
from salure_tfx_extensions.types.component_specs import BaseSpec


class BaseComponent(base_component.BaseComponent):
    """A component that loads in files, stored in tf.example format, and spits those out again.

    This component is meant as a boilerplate for other custom TFX components.
    """

    SPEC_CLASS = BaseSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(self,
                 examples: types.Channel,
                 input_config: Optional[Union[example_gen_pb2.Input, Dict[Text, Any]]] = None,
                 instance_name: Optional[Text] = None):
        """
        This code gets run, when you define the component in the pipeline definition
        :param examples: A ChannelParameter representing tf Examples
        """

        input_config = input_config or utils.make_default_input_config()
        output_config = example_gen_pb2.Output()

        artifact = standard_artifacts.Examples()
        artifact.split_names = artifact_utils.encode_split_names(
            utils.generate_output_split_names(input_config, output_config)
        )
        example_artifacts = channel_utils.as_channel([artifact])


        spec = BaseSpec(
            input_config=input_config,
            output_config=output_config,
            examples=examples,
            output_examples=example_artifacts
        )
        super(BaseComponent, self).__init__(spec=spec, instance_name=instance_name)
