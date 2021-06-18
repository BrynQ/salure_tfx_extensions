"""Component to Analyze the results"""

from tfx import types
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from salure_tfx_extensions.components.mysql_pusher_percentile import executor
from salure_tfx_extensions.types.component_specs import MySQLPusherSpec
from typing import Optional, Text


class MySQLPusher(base_component.BaseComponent):
    """
    A component that loads in Inference results and calculated percentile values,
    return the input files with a label to the results.
    """

    SPEC_CLASS = MySQLPusherSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(self,
                 inference_result: types.Channel,
                 percentile_values:types.Channel,
                 connection_config,
                 instance_name: Optional[Text] = None):

        spec = MySQLPusherSpec(
            inference_result=inference_result,
            percentile_values=percentile_values,
            connection_config=connection_config,
        )

        super(MySQLPusher, self).__init__(spec=spec, instance_name=instance_name)