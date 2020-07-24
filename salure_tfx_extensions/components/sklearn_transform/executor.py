import dill
import base64
from typing import Any, Dict, List, Text, Type
from tfx import types
from salure_tfx_extensions.components.base.sklearn_base.executor import SKLearnBaseExecutor
import apache_beam as beam
from apache_beam import pvalue

PREPROCESSOR_PICKLE_KEY = 'preprocessor_pickle'
TRANSFORM_PIPELINE_KEY = 'transform_pipeline'
_TELEMETRY_DESCRIPTORS = ['SKLearnTransform']
PIPELINE_FILE_NAME = 'pipeline.pickle'


class _FitPreprocessor(beam.PTransform):
    def __init__(self, preprocessor, exec_properties):
        self.preprocessor = preprocessor
        super(_FitPreprocessor, self).__init__()

    def expand(self, train_df):
        return (
            train_df
            | 'Fitting Preprocessor' >> beam.FlatMap(
                _FitPreprocessor._fit_sklearn_preprocessor,
                pvalue.AsSingleton(self.preprocessor)))

    @staticmethod
    def _fit_sklearn_preprocessor(df, preprocessor):
        preprocessor.fit(df)
        yield preprocessor


class _ApplyPreprocessor(beam.PTransform):
    def __init__(self, preprocessor, exec_properties):
        self.preprocessor = preprocessor
        super(_ApplyPreprocessor, self).__init__()

    def expand(self, df):
        return (
            df
            | 'Applying Preprocessor Transformation' >> beam.FlatMap(
                _ApplyPreprocessor._apply_sklearn_preprocessor,
                pvalue.AsSingleton(self.preprocessor)))

    @staticmethod
    def _apply_sklearn_preprocessor(df, preprocessor):
        yield preprocessor.transform(df)


class Executor(SKLearnBaseExecutor):
    def get_sklearn_object(self, input_dict: Dict[Text, List[types.Artifact]],
                           output_dict: Dict[Text, List[types.Artifact]],
                           exec_properties: Dict[Text, Any]) -> beam.PTransform:
        return dill.loads(base64.decodebytes(exec_properties[PREPROCESSOR_PICKLE_KEY].encode('utf-8')))

    @property
    def GetFitSKLearnTransform(self) -> Type[beam.PTransform]:
        return _FitPreprocessor

    @property
    def GetApplySKLearnTransform(self) -> Type[beam.PTransform]:
        return _ApplyPreprocessor

    @property
    def telemetry_descriptors(self):
        return _TELEMETRY_DESCRIPTORS

    @property
    def model_output_artifact_key(self):
        return TRANSFORM_PIPELINE_KEY

    @property
    def sklearn_file_name(self):
        return PIPELINE_FILE_NAME
