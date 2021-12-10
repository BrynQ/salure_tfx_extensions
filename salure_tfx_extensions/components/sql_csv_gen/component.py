"""Salure TFX MySQLExampleGen Component"""

from salure_tfx_extensions.components.sql_csv_gen import executor
from salure_tfx_extensions.proto import mysql_config_pb2
from typing import Optional, Text
from tfx.types import Channel
from tfx.types.standard_component_specs import QueryBasedExampleGenSpec
from tfx.components.example_gen import component
from tfx.components.example_gen import utils
from tfx.dsl.components.base import executor_spec
from tfx.proto import example_gen_pb2


class SQLCSVExampleGen(component.QueryBasedExampleGen):

    SPEC_CLASS = QueryBasedExampleGenSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(self,
                 conn_config: mysql_config_pb2.MySQLConnConfig,
                 query: Optional[Text] = None,
                 input_config: Optional[example_gen_pb2.Input] = None,
                 output_config: Optional[example_gen_pb2.Output] = None,
                 example_artifacts: Optional[Channel] = None,
                 instance_name: Optional[Text] = None):

        if bool(query) == bool(input_config):
            raise ValueError('Exactly one of query and input_config should be set.')
        if not bool(conn_config.host):
            raise ValueError('Required host field in connection config should be set.')

        input_config = input_config or utils.make_default_input_config(query)
        custom_config = example_gen_pb2.CustomConfig()
        custom_config.custom_config.Pack(conn_config)

        output_config = output_config or utils.make_default_output_config(input_config)

        super(SQLCSVExampleGen, self).__init__(
            input_config=input_config,
            output_config=output_config,
            custom_config=custom_config,
            example_artifacts=example_artifacts,
            instance_name=instance_name)