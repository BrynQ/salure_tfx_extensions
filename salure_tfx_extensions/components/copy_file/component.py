from tfx.components.base.base_component import BaseComponent
from tfx.dsl.components.base import executor_spec
from typing import Optional, Text
from salure_tfx_extensions.components.copy_file import executor
from salure_tfx_extensions.types.component_specs import CopyFileSpec


class CopyFileComponent(BaseComponent):

    SPEC_CLASS = CopyFileSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(self, input_path: Optional[Text] = None,
                 output_path: Optional[Text] = None,
                 instance_name: Optional[Text] = None):

        spec = CopyFileSpec(input_path = input_path, output_path=output_path)
        super(CopyFileComponent, self).__init__(spec=spec, instance_name=instance_name)