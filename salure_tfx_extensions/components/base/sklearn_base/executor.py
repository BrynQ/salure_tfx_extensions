import abc
from abc import abstractmethod
import os
import absl
from typing import Any, Dict, List, Text, Type
from tfx import types
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import io_utils
import tensorflow_transform.beam as tft_beam
from salure_tfx_extensions.utils import sklearn_utils
import apache_beam as beam
from six import with_metaclass

EXAMPLES_KEY = 'examples'
SCHEMA_KEY = 'schema'
TRANSFORMED_EXAMPLES_KEY = 'transformed_examples'


class SKLearnBaseExecutor(with_metaclass(abc.ABCMeta, base_executor.BaseExecutor)):
    """Executor for the SKLearnTransform Component
    Reads in Examples, and extracts a Pipeline object from a module file.
    It fits the pipeline, and writes the fit pipeline and the transformed examples to file"""

    @abc.abstractmethod
    def get_sklearn_object(self,
                           input_dict: Dict[Text, List[types.Artifact]],
                           output_dict: Dict[Text, List[types.Artifact]],
                           exec_properties: Dict[Text, Any]) -> beam.PTransform:
        """Returns a Singleton pcoll containing the sklearn object"""
        pass

    @property
    @abstractmethod
    def GetFitSKLearnTransform(self) -> Type[beam.PTransform]:
        """Should return a PTransform that can take in a DF and a singleton sklearn Model
        The PTransform should return the fit SKLearn object

        The init takes in a singleton sklearn Model, and exec_properties"""
        pass

    @property
    @abstractmethod
    def GetApplySKLearnTransform(self) -> Type[beam.PTransform]:
        """Should return a PTransform that can take in a DF and a singleton sklearn Model
        The PTransform should return the transformed DF

        The init takes in a singleton sklearn Model, and exec_properties"""
        pass

    @property
    @abstractmethod
    def telemetry_descriptors(self):
        """If unknown you can leave this None"""
        return None

    @property
    @abstractmethod
    def model_output_artifact_key(self):
        """The key for output_dict, containing the sklearn object output Artifact"""
        pass

    @property
    @abstractmethod
    def sklearn_file_name(self):
        """The file name of the file to write the sklearn object to"""
        pass

    def Do(self,
           input_dict: Dict[Text, List[types.Artifact]],
           output_dict: Dict[Text, List[types.Artifact]],
           exec_properties: Dict[Text, Any]) -> None:
        """
        Args:
          input_dict:
            - examples: Tensorflow Examples
          exec_properties:
            - preprocessor_pickle: A pickle string of the preprocessor
            - preprocessor_pipeline_name: The name of the pipeline object in the specified module file
          output_dict:
            - transformed_examples: The transformed Tensorflow Examples
            - transform_pipeline: A trained SKLearn Pipeline
        """

        self._log_startup(input_dict, output_dict, exec_properties)

        if not (len(input_dict[EXAMPLES_KEY]) == 1):
            raise ValueError('input_dict[{}] should only contain one artifact'.format(EXAMPLES_KEY))

        examples_artifact = input_dict[EXAMPLES_KEY][0]
        examples_splits = artifact_utils.decode_split_names(examples_artifact.split_names)

        train_and_eval_split = ('train' in examples_splits and 'eval' in examples_splits)
        single_split = ('single_split' in examples_artifact.uri)

        if train_and_eval_split == single_split:
            raise ValueError('Couldn\'t determine which input split to fit the pipeline on. '
                             'Exactly one split between \'train\' and \'single_split\' should be specified.')

        train_split = 'train' if train_and_eval_split else 'single_split'

        train_uri = os.path.join(examples_artifact.uri, train_split)

        # Load in the schema
        schema_path = io_utils.get_only_uri_in_dir(
            artifact_utils.get_single_uri(input_dict[SCHEMA_KEY]))
        schema = io_utils.SchemaReader().read(schema_path)

        sklearn_object = self.get_sklearn_object(input_dict, output_dict, exec_properties)

        absl.logging.info('sklearn loaded in: {}'.format(sklearn_object))

        with self._make_beam_pipeline() as pipeline:
            with tft_beam.Context(
                    use_deep_copy_optimization=True):
                absl.logging.info('Loading Training Examples')
                train_input_uri = io_utils.all_files_pattern(train_uri)
                preprocessor_output_uri = artifact_utils.get_single_uri(output_dict[self.model_output_artifact_key])

                absl.logging.info(input_dict)
                absl.logging.info(output_dict)

                training_data = pipeline | 'Read Train Data' >> sklearn_utils.ReadTFRecordToPandas(
                    file_pattern=train_input_uri,
                    schema=schema,
                    split_name='Train',  # Is just for naming the beam operations
                    telemetry_descriptors=self.telemetry_descriptors
                )

                preprocessor_pcoll = pipeline | beam.Create([sklearn_object])

                fit_sklearn_processor = training_data | 'Fit SKLearn Processor' >> self.GetFitSKLearnTransform(
                    preprocessor_pcoll,
                    exec_properties)

                transformed_df = training_data | 'Transform Training Data' >> self.GetApplySKLearnTransform(
                    fit_sklearn_processor,
                    exec_properties)

                fit_sklearn_processor | sklearn_utils.WriteSKLearnModelToFile(
                    os.path.join(preprocessor_output_uri, self.sklearn_file_name))

                transformed_df | 'Write Train Data to File' >> sklearn_utils.WriteDataFrame(
                    os.path.join(output_dict[TRANSFORMED_EXAMPLES_KEY][0].uri, train_split))

                if train_and_eval_split:
                    test_split = 'eval'
                    test_uri = os.path.join(examples_artifact.uri, test_split)
                    test_input_uri = io_utils.all_files_pattern(test_uri)

                    test_data = pipeline | 'Read Test Data' >> sklearn_utils.ReadTFRecordToPandas(
                        file_pattern=test_input_uri,
                        schema=schema,
                        split_name='Test',  # Is just for naming the beam operations
                        telemetry_descriptors=self.telemetry_descriptors
                    )

                    transformed_test_data = test_data | 'Transform Test Data' >> self.GetApplySKLearnTransform(
                        fit_sklearn_processor,
                        exec_properties)

                    transformed_test_data | 'Write Test Data to File' >> sklearn_utils.WriteDataFrame(
                        os.path.join(output_dict[TRANSFORMED_EXAMPLES_KEY][0].uri, test_split))
