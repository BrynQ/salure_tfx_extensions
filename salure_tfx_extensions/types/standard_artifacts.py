"""The Artifact types for salure_tfx_extensions"""

from tfx.types.artifact import Artifact


class SKLearnModel(Artifact):
    TYPE_NAME = 'SKLearnModel'


class SKLearnPrepocessor(Artifact):
    TYPE_NAME = 'SKLearnPreprocessor'
