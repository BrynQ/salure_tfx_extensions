from typing import Optional, Text, Dict, Any, Union

from tfx import types
# from tfx.components.base import base_component
# from tfx.components.base import executor_spec
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from tfx.types import standard_artifacts
from tfx.types import artifact_utils
from tfx.types import channel_utils
from salure_tfx_extensions.components.mysql_pusher import executor
from salure_tfx_extensions.types.component_specs import MySQLPusherSpec


class MySQLPusher(base_component.BaseComponent):
    """A component that loads in files, stored in tf.example format, and spits those out again.

    This component is meant as a boilerplate for other custom TFX components.
    """

    SPEC_CLASS = MySQLPusherSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(self,
                 inference_result: types.Channel,
                 module_file: str,
                 instance_name: Optional[Text] = None):
        """
        This code gets run, when you define the component in the pipeline definition
        :param examples: A ChannelParameter representing tf Examples
        """

        spec = MySQLPusherSpec(
            inference_result=inference_result,
            module_file=module_file
        )

        super(MySQLPusher, self).__init__(spec=spec, instance_name=instance_name)
