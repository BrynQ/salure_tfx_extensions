"""A trainer component for SKLearn models"""

import os
import dill
import base64
import joblib
import absl
import apache_beam as beam
from apache_beam import pvalue
import tensorflow as tf
from typing import Any, Dict, List, Text, Tuple, Type
from tfx import types
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.utils import path_utils
from salure_tfx_extensions.utils import example_parsing_utils
from salure_tfx_extensions.utils import sklearn_utils
from salure_tfx_extensions.components.base.sklearn_base.executor import SKLearnBaseExecutor
from salure_tfx_extensions import constants

from sklearn.base import BaseEstimator

# from tfx_bsl.tfxio import tf_example_record

MODEL_PICKLE_KEY = 'model_pickle'
MODEL_KEY = 'model'
LABEL_NAME_KEY = 'label_name'
_TELEMETRY_DESCRIPTORS = ['SKLearnTrainer']
MODEL_FILE_NAME = 'model.pickle'


# class Executor(base_executor.BaseExecutor):
#     """Executor for the SKLearnTrainer Component
#
#     Takes in Examples, parses them, and trains an SKLearnModel on it
#     """
#
#     # TODO: NOT DONE AT ALL
#     def Do(self, input_dict: Dict[Text, List[types.Artifact]],
#            output_dict: Dict[Text, List[types.Artifact]],
#            exec_properties: Dict[Text, Any]) -> None:
#         """
#         Args:
#           input_dict:
#             - examples: Examples used for training, must include 'train' and 'eval' splits
#           output_dict:
#             - model: The trained SKLearnModel
#           exec_properties:
#             - model_pickle: A bytestring contain a pickled SKLearn model
#
#         """
#         self._log_startup(input_dict, output_dict, exec_properties)
#
#         if 'examples' not in input_dict:
#             raise ValueError('\'examples\' is missing in input dict')
#         if 'model_pickle' not in exec_properties:
#             raise ValueError('\'model_pickle\' is missing in exec_properties')
#         if 'supervised' not in exec_properties:
#             raise ValueError('\'supervised\' is missing in exec_properties')
#         if 'model' not in output_dict:
#             raise ValueError('\'model\' is missing in output_dict')
#
#         model = pickle.loads(exec_properties['model_pickle'])
#         is_supervised = exec_properties['supervised']
#
#         absl.logging.info(model)
#
#         split_uris = []
#
#         if not len(input_dict[EXAMPLES_KEY]) == 1:
#             raise ValueError('input_dict[{}] should contain only 1 artifact'.format(
#                 EXAMPLES_KEY))
#
#         artifact = input_dict[EXAMPLES_KEY][0]
#         splits = artifact_utils.decode_split_names(artifact.split_names)
#
#         train_uri, eval_uri = example_parsing_utils.get_train_and_eval_uris(artifact, splits)
#
#         with self._make_beam_pipeline() as pipeline:
#             training_data = (
#                 pipeline
#                 | 'ReadTrainingExamplesFromTFRecord' >> beam.io.ReadFromTFRecord(
#                     file_pattern=train_uri)
#                 | 'ParseTrainingExamples' >> beam.Map(tf.train.Example.FromString))
#
#             # training_data_rows is PCollection of List[Any]
#             training_data_rows = (
#                 training_data
#                 | 'Training Example to rows' >> beam.Map(
#                     example_parsing_utils.example_to_list)
#                 | 'Aggregating training rows' >> beam.CombineGlobally(
#                     example_parsing_utils.CombineFeatureLists()))
#
#             model_pcoll = (
#                 training_data_rows
#                 | 'Rows To Numpy' >> beam.Map(example_parsing_utils.to_numpy_ndarray)
#                 | 'Train SKLearnModel' >> beam.ParDo(TrainSKLearnModel(model))
#                 | 'Write Model to file' >> sklearn_utils.WriteSKLearnModelToFile(output_dict['model'] +
#                                                                                  constants.DEFAULT_SKLEARN_MODEL_NAME))
#
#             # TODO: Support label key in the tf.examples
#             # TODO: Make it possible to train unsupervised models
#             # TODO: Use CombineFn to aggregate tf.examples into single 2D list
#                 # tip: parse tf.Example to python list using stfxe.utils.example_parsing
#
class _FitModel(beam.PTransform):
    def __init__(self, model, exec_properties):
        self.model = model
        self.df_label_name = exec_properties[LABEL_NAME_KEY]
        super(_FitModel, self).__init__()

    def expand(self, train_df):
        return (
            train_df
            | 'Fitting Model' >> beam.FlatMap(
                self._train_model,
                pvalue.AsSingleton(self.model)))

    def _train_model(self, df, model):
        if self.df_label_name:
            labels = df.pop(self.df_label_name)
            model.fit(df, labels)
            return model

        model.fit(df)
        return model


class _ApplyModel(beam.PTransform):
    def __init__(self, model, exec_properties):
        self.model = model
        super(_ApplyModel, self).__init__()

    def expand(self, df):
        return (
            df
            | 'Applying Model Prediction Transformation' >> beam.FlatMap(
                _ApplyModel._apply_sklearn_preprocessor,
                pvalue.AsSingleton(self.model)))

    @staticmethod
    def _apply_sklearn_preprocessor(df, model):
        yield model.predict(df)


class Executor(SKLearnBaseExecutor):
    def get_sklearn_object(self, input_dict: Dict[Text, List[types.Artifact]],
                           output_dict: Dict[Text, List[types.Artifact]],
                           exec_properties: Dict[Text, Any]) -> beam.PTransform:
        return dill.loads(base64.decodebytes(exec_properties[MODEL_PICKLE_KEY].encode('utf-8')))

    @property
    def GetFitSKLearnTransform(self) -> Type[beam.PTransform]:
        pass

    @property
    def GetApplySKLearnTransform(self) -> Type[beam.PTransform]:
        return _ApplyModel

    @property
    def telemetry_descriptors(self):
        return _TELEMETRY_DESCRIPTORS

    @property
    def model_output_artifact_key(self):
        return MODEL_KEY

    @property
    def sklearn_file_name(self):
        return MODEL_FILE_NAME
