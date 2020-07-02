"""Helper functions for handling sklearn models"""

import apache_beam as beam
from salure_tfx_extensions import constants
import joblib


class WriteSKLearnModelToFile(beam.PTransform):
    def __init__(self, file_path):
        self.file_path = file_path

    def expand(self, model, *args, **kwargs):
        joblib.dump(model, self.file_path + constants.DEFAULT_SKLEARN_MODEL_NAME)
        return model
