from tensorflow_transform.tf_metadata import schema_utils
import tensorflow as tf


def schema_to_feature_spec(schema):
    return schema_utils.schema_as_feature_spec(schema).feature_spec


def tfrecord_reader_fn(filenames, compression_type='GZIP'):
    """Small utility returning a record reader that can read gzip'ed files."""
    return tf.data.TFRecordDataset(
        filenames,
        compression_type=compression_type)


def create_feature_columns_fn(numerical_column_names, categorical_column_names, categorical_dtype=tf.int64):
    def get_feature_columns(tf_transform_output):
        """Returns the FeatureColumns for the model.

        Args:
            tf_transform_output: A `TFTransformOutput` object.

        Returns:
            A list of FeatureColumns.
        """
        # Wrap scalars as real valued columns.
        real_valued_columns = [tf.feature_column.numeric_column(key, shape=())
                               for key in numerical_column_names]

        # Wrap categorical columns.
        categorical_columns = [
            tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_file(
                    key=key,
                    dtype=categorical_dtype,
                    vocabulary_file=tf_transform_output.vocabulary_file_by_name(
                        vocab_filename=key)))
            for key in categorical_column_names]

        return real_valued_columns + categorical_columns

    return get_feature_columns
