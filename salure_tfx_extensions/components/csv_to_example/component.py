from salure_tfx_extensions.components.csv_to_example import executor
from salure_tfx_extensions.types.component_specs import CsvToExampleSpec
from tfx.types import standard_artifacts, channel_utils
from tfx.components.base.base_component import BaseComponent
from tfx.dsl.components.base import executor_spec
from typing import Optional, Text


class CsvToExampleComponent(BaseComponent):

    SPEC_CLASS = CsvToExampleSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(self, input_path: Optional[Text] =None,
                 feature_description: Optional[Text] =None,
                 instance_name: Optional[Text] = None):
        examples=channel_utils.as_channel([standard_artifacts.Examples()])
        spec = CsvToExampleSpec(input_path = input_path, feature_description=json_utils.dumps(feature_description), examples=examples)
        super(CsvToExampleComponent, self).__init__(spec=spec, instance_name=instance_name)