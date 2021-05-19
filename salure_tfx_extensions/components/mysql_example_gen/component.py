"""Salure TFX MySQLExampleGen Component"""

from typing import Optional, Text
from tfx.types import Channel
from tfx.types.standard_component_specs import QueryBasedExampleGenSpec
from tfx.components.example_gen import component
from tfx.components.example_gen import utils
from tfx.dsl.components.base import executor_spec
from tfx.proto import example_gen_pb2
from salure_tfx_extensions.components.mysql_example_gen import executor
from salure_tfx_extensions.proto import mysql_config_pb2


class MySQLExampleGen(component.QueryBasedExampleGen):

    SPEC_CLASS = QueryBasedExampleGenSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(self,
                 conn_config: mysql_config_pb2.MySQLConnConfig,
                 query: Optional[Text] = None,
                 input_config: Optional[example_gen_pb2.Input] = None,
                 output_config: Optional[example_gen_pb2.Output] = None,
                 example_artifacts: Optional[Channel] = None,
                 instance_name: Optional[Text] = None):
        """Constructs a MySQLExampleGen component.

        Args:
          conn_config: Parameters for PyMySQL connection client.
          query: MySQL string, query result will be treated as a single split,
            can be overwritten by input_config.
          input_config: An example_gen_pb2.Input instance with Split.pattern as
            sql string. If set, it overwrites the 'query' arg, and allows
            different queries per split.
          output_config: An example_gen_pb2.Output instance, providing output
            configuration. If unset, default splits will be 'train' and 'eval' with
            size 2:1.
          example_artifacts: Optional channel of 'ExamplesPath' for output train and
            eval examples.
          name: Optional unique name. Necessary if multiple PrestoExampleGen
            components are declared in the same pipeline.
        Raises:
          ValueError: Only one of query and input_config should be set. Or
          required host field in connection_config should be set.
        """
        if bool(query) == bool(input_config):
            raise ValueError('Exactly one of query and input_config should be set.')
        if not bool(conn_config.host):
            raise ValueError('Required host field in connection config should be set.')

        input_config = input_config or utils.make_default_input_config(query)
        custom_config = example_gen_pb2.CustomConfig()
        custom_config.custom_config.Pack(conn_config)
        output_config = output_config or utils.make_default_output_config(input_config)

        super(MySQLExampleGen, self).__init__(
            input_config=input_config,
            output_config=output_config,
            custom_config=custom_config,
            example_artifacts=example_artifacts,
            instance_name=instance_name)
