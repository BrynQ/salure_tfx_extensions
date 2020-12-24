from typing import List, Union, Optional
import abc
from kfp import dsl
from tfx.components.base.base_component import BaseComponent


class BaseDeployment(abc.ABC):

    # @abc.abstractmethod
    # @property
    def resource_op(self) -> dsl.ResourceOp:
        """Generates a kfp dsl ResourceOp.
        Must be initialized in the runtime of the function passed to the kfp compiler
        for the compiler to register it."""
        return NotImplemented

    # @abc.abstractmethod
    @property
    def dependents(self) -> Optional[List[BaseComponent]]:
        return NotImplemented
