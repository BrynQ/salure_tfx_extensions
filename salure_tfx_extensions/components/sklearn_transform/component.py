from typing import Optional, Text, Dict, Any, Union

import dill
import base64

from tfx import types
from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.types import standard_artifacts
from tfx.types import artifact_utils
from tfx.proto import example_gen_pb2
from tfx.components.example_gen import utils
from tfx.types import channel_utils
from salure_tfx_extensions.components.sklearn_transform import executor
from salure_tfx_extensions.types.component_specs import SKLearnTransformSpec
import salure_tfx_extensions.types.standard_artifacts as stfxe_artifacts

from sklearn.pipeline import Pipeline


class SKLearnTransform(base_component.BaseComponent):

    """The SKLearnTransform component reads an SKLearn Pipeline object from a module file,
    Fits the pipeline, and stores it as an artifact
    """

    SPEC_CLASS = SKLearnTransformSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(self,
                 examples: types.Channel,
                 schema: types.Channel,
                 sklearn_pipeline: Pipeline = None,
                 instance_name: Optional[Text] = None):

        # Pickle must be a string to put it in json
        preprocessor_pickle = base64.encodebytes(dill.dumps(sklearn_pipeline)).decode('utf-8')

        preprocessor_artifact = channel_utils.as_channel([stfxe_artifacts.SKLearnPrepocessor()])
        transformed_examples_artifact = channel_utils.as_channel([standard_artifacts.Examples()])

        spec = SKLearnTransformSpec(
            examples=examples,
            schema=schema,
            preprocessor_pickle=preprocessor_pickle,
            transformed_examples=transformed_examples_artifact,
            transform_pipeline=preprocessor_artifact
        )

        super(SKLearnTransform, self).__init__(spec=spec, instance_name=instance_name)

