"""Helper functions for parsing and handling tf.Examples"""

from typing import Text, List, Any, Union
import tensorflow as tf
import absl


def example_to_list(example: tf.train.Example) -> List[Union[Text, int, float]]:
    # TODO
    return NotImplemented
