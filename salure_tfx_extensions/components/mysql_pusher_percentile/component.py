from typing import Optional, Text
from tfx import types
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from salure_tfx_extensions.types.component_specs import MySQLPusherSpec
from salure_tfx_extensions.components.mysql_pusher_percentile.executor import Executor

class MySQLPusher(base_component.BaseComponent):
    """
    A component that loads in inference results and calculated percentile values,
    then uploads the input files with a label to the results to a specified table
    in a MySQL database.
    """

    SPEC_CLASS = MySQLPusherSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(Executor)

    def __init__(self,
                 inference_result: types.Channel,
                 percentile_values: types.Channel,
                 connection_config,
                 table_name,
                 inference_id,
                 instance_name: Optional[Text] = None):

        spec = MySQLPusherSpec(
            inference_result=inference_result,
            percentile_values=percentile_values,
            connection_config=connection_config,
            table_name=table_name,
            inference_id=inference_id
        )

        super(MySQLPusher, self).__init__(spec=spec, instance_name=instance_name)
