from typing import Optional, Text

from tfx import types
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from salure_tfx_extensions.components.mysql_pusher import executor
from salure_tfx_extensions.types.component_specs import MySQLPusherSpec


class MySQLPusher(base_component.BaseComponent):
    """
    A component that loads in files, stored in tf.example format, and spits those out again.
    """

    SPEC_CLASS = MySQLPusherSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(self,
                 inference_result: types.Channel,
                 connection_config,
                 instance_name: Optional[Text] = None):

        spec = MySQLPusherSpec(
            inference_result=inference_result,
            connection_config=connection_config,
        )

        super(MySQLPusher, self).__init__(spec=spec, instance_name=instance_name)
