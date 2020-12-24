"""A trainer component for SKLearn models"""

from typing import Optional, Text, Dict, Any, Union

import dill
import base64
from tfx import types
from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.types import standard_artifacts
from salure_tfx_extensions.types import standard_artifacts as stfxe_artifacts
from tfx.types import artifact_utils
from tfx.proto import example_gen_pb2
from tfx.components.example_gen import utils
from tfx.types import channel_utils
from salure_tfx_extensions.components.sklearn_trainer import executor
from salure_tfx_extensions.types.component_specs import SKLearnTrainerSpec
from sklearn.base import BaseEstimator


class SKLearnTrainer(base_component.BaseComponent):
    """A component which trains SKLearn models using their 'fit' function"""

    SPEC_CLASS = SKLearnTrainerSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(self,
                 examples: types.Channel,
                 schema: types.Channel,
                 sklearn_model: BaseEstimator,  # should have 'fit' and 'predict'
                 label_name: Optional[str] = None,
                 instance_name: Optional[Text] = None):
        """

        :param examples: A TFX Channel of type 'Examples'
        :param sklearn_model: The actual SKLearn Model to be pickled and to be executed
        :param label_name: The name of the column that contains the labels, unsupervised if None
        """
        if label_name is None:
            label_name = ''

        # TODO: Allow for transformed inputs, and transformation input graph
        model_pickle = base64.encodebytes(dill.dumps(sklearn_model)).decode('utf-8')

        model_artifact = channel_utils.as_channel([stfxe_artifacts.SKLearnModel()])
        transformed_examples_artifact = channel_utils.as_channel([standard_artifacts.Examples()])

        spec = SKLearnTrainerSpec(
            examples=examples,
            schema=schema,
            model_pickle=model_pickle,
            label_name=label_name,
            transformed_examples=transformed_examples_artifact,
            model=model_artifact,
        )

        super(SKLearnTrainer, self).__init__(
            spec=spec,
            instance_name=instance_name,
            # enable_cache=enable_cache,
        )
