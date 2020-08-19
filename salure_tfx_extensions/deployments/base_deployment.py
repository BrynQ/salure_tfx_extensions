from typing import List, Union
import abc
from kfp import dsl
from tfx.components.base.base_component import BaseComponent


class BaseDeployment(abc.ABC):

    @abc.abstractmethod
    @property
    def resource_op(self) -> dsl.ResourceOp:
        return NotImplemented

    @abc.abstractmethod
    @property
    def dependents(self) -> List[BaseComponent]:
        return NotImplemented
