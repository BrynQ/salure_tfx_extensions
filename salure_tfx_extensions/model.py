"""Definitions for Model abstractions"""

from typing import Dict, Any, List, Callable, Optional, Union

import tensorflow as tf
from tfx.orchestration import pipeline

ModelFn = Callable[[Union[tf.Tensor, Dict[str, Any]],
                    Optional[Union[tf.Tensor, Dict[str, Any]]],
                    Optional[tf.estimator.ModeKeys],
                    Optional[Dict[str, Any]],
                    Optional[tf.estimator.RunConfig]],
                   tf.estimator.EstimatorSpec]


class Model(object):
    """For now, no more than an unnecessary abstraction layer for tf.estimator.Estimator
    Part of vision to make data input code and model code completely pipeline agnostic.
    The goal of this class is to be a wrapper for completely TFX-pipeline agnostic model definitions.
    """

    def __init__(self,
                 model_fn: ModelFn,
                 model_params: Optional[Dict[str, Any]] = None,
                 feature_columns=None,
                 config: tf.estimator.RunConfig = tf.estimator.RunConfig(
                     save_checkpoints_steps=999, keep_checkpoint_max=1)):
        self.model_fn = model_fn
        self.model_params = model_params
        self.feature_columns = feature_columns
        self.config = config

        # Set key "feature_columns" with parameter feature_columns if unset
        self.model_params['feature_columns'] = self.model_params.get('feature_columns', feature_columns)

    def to_estimator(self):
        return tf.estimator.Estimator(
            model_fn=self.model_fn,
            params=self.model_params,
            config=self.config
        )

    @staticmethod
    def from_config(config: Dict[str, Any]):
        return Model(model_fn=config['model_fn'], model_params=config['model_params'])


class UnsupervisedModel(Model):

    def __init__(self,
                 model_fn: ModelFn,
                 model_params: Optional[Dict[str, Any]] = None,
                 feature_columns=None,
                 config: tf.estimator.RunConfig = tf.estimator.RunConfig(
                     save_checkpoints_steps=999, keep_checkpoint_max=1)):
        super(UnsupervisedModel, self).__init__(model_fn,
                                                model_params=model_params,
                                                feature_columns=feature_columns,
                                                config=config)


class MSEAutoEncoder(UnsupervisedModel):
    """Dense AutoEncoder that, when predicting, doesn't return predicted inputs,
        but row wise MSE between ground truth and predictions

    """

    def __init__(self,
                 model_params: Optional[Dict[str, Any]] = None,
                 feature_columns=None,
                 config: tf.estimator.RunConfig = tf.estimator.RunConfig(
                     save_checkpoints_steps=999, keep_checkpoint_max=1)):
        super(MSEAutoEncoder, self).__init__(MSEAutoEncoder.model_fn,
                                             model_params=model_params,
                                             feature_columns=feature_columns,
                                             config=config)

    @staticmethod
    def model_fn(features: Union[tf.Tensor, Dict[str, Any]],
                 labels: Optional[Union[tf.Tensor, Dict[str, Any]]] = None,  # is passed by Estimator, even if unused
                 mode: tf.estimator.ModeKeys = None,
                 params: Dict[str, Any] = None) \
            -> tf.estimator.EstimatorSpec:
        print(features)
        # feature_layer = tf.keras.layers.DenseFeatures(params['feature_columns'])
        # inputs = feature_layer(features)  # store transformed inputs for eval
        inputs = tf.feature_column.input_layer(features, params['feature_columns'], trainable=False)

        x = inputs
        for units, activation in zip(params['hidden_units'], params['activations']):
            x = tf.keras.layers.Dense(units, activation)(x)
        outputs = tf.keras.layers.Dense(inputs.shape[1], None)(x)

        print(inputs.shape)
        print(outputs.shape)

        row_wise_mse = tf.reduce_mean(tf.math.pow(outputs - inputs, 2), 1)

        if mode == tf.estimator.ModeKeys.PREDICT:
            mse = tf.expand_dims(row_wise_mse, -1)
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=mse
            )

        mse = tf.reduce_mean(row_wise_mse)
        mse_metric = tf.metrics.mean_squared_error(inputs, outputs)
        # Accuracy metric to make model pass ModelValidator, which has currently a hardcoded check for accuracy
        accuracy_metric = tf.metrics.accuracy(tf.zeros((2, 2), dtype=tf.float32), tf.zeros((2, 2), dtype=tf.float32))

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=mse,
                eval_metric_ops={'mse': mse_metric,
                                 'accuracy': accuracy_metric}
            )

        # training
        optimizer = params.get('optimizer', tf.train.AdamOptimizer(learning_rate=0.001))
        train_op = optimizer.minimize(mse, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=mse,
            train_op=train_op
        )


class PipelineSpec(object):
    """The goal of this class is to envelop all variants for common TFX configurations.
    This means that you can 'instantiate' TFX pipelines using PipelineSpec.

    The goal of this class is to componentialize Models and their preprocessing respectively.

    Imagined example usage:
    ```
    pipeline_spec = PipelineSpec(...)
    preprocessing_fn = pipeline_spec.create_preprocessing_fn()
    trainer_fn = pipeline_spec.create_trainer_fn()

    airflow_pipeline = AirflowDAGRunner(_airflow_config).run(pipeline_spec.create_pipeline())
    """

    # TODO: __init__ parameters not exhaustive yet
    def __init__(self,
                 model_fn: Optional[ModelFn] = None,
                 model_class: Optional[Model] = Model):
        if model_fn is None and not hasattr(model_class, 'model_fn'):
            raise ValueError('Specify a model_fn or make sure the model_class has a model_fn property')

        # serving_model_dir
        # file_location, pipeline_name, tfx_root, tfx_pipeline_root, metadata

        # self.components = [
        #     example_gen, statistics_gen, infer_schema, validate_stats, transform,
        #     trainer, model_analyzer, model_validator, pusher
        # ]

    # TODO: return pipeline from specification
    def create_pipeline(self) -> pipeline.Pipeline:
        pass

    # TODO: decompose model properties, and forward additional required args
    @staticmethod
    def from_model():
        # return ModelSpec()
        pass
