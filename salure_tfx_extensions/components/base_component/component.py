from typing import Optional, Text

from tfx import types
from tfx.components.base import base_component
from tfx.components.base import executor_spec
from salure_tfx_extensions.components.base_component import executor
from tfx.types import standard_artifacts
from salure_tfx_extensions.types.component_specs import BaseSpec


class BaseComponent(base_component.BaseComponent):
    """A component that loads in files, stored in tf.example format, and spits those out again.

    This component is meant as a boilerplate for other custom TFX components.
    """

    SPEC_CLASS = BaseSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(self,
                 examples: types.Channel,
                 instance_name: Optional[Text] = None):
        """
        This code gets run, when you define the component in the pipeline definition
        :param examples: A ChannelParameter representing tf Examples
        """
        spec = BaseSpec(examples=examples)
        super(BaseComponent, self).__init__(spec=spec, instance_name=instance_name)
