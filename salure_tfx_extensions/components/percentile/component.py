"""Custom TFX Component to Calculate the Threshold for Autoencoder"""

from tfx.dsl.components.base import executor_spec
from tfx import types
from typing import Optional, Text
from tfx.types import standard_artifacts, channel_utils
from tfx.components.base.base_component import BaseComponent
from salure_tfx_extensions.components.percentile import executor
from salure_tfx_extensions.types.component_specs import PercentileComponentSpec


class PercentileComponent(BaseComponent):

    SPEC_CLASS = PercentileComponentSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)


    def __init__(self, inference_result: types.Channel,
                 num_quantiles: Text,
                 percentile_values: types.Channel = None,
                 instance_name: Optional[Text] = None):

        if not percentile_values:
            examples_artifact = standard_artifacts.Examples()
            percentile_values = channel_utils.as_channel([examples_artifact])

        spec = PercentileComponentSpec(inference_result=inference_result,
                                       num_quantiles = num_quantiles,
                                       percentile_values=percentile_values)

        super(PercentileComponent, self).__init__(spec=spec, instance_name=instance_name)