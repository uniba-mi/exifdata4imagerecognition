from abc import ABC, abstractmethod
from typing import Any, List, TypeVar, Generic

T = TypeVar('T')
O = TypeVar('O')

class Transformer(ABC, Generic[T, O]):
    """ A tranformer applies a specific transformation function to an object and returns a new (or the same) object of any type. """
    
    @abstractmethod
    def transform(self, data: T) -> O:
        pass


class TransformError(Exception):
    pass


class CompositeTransformer(Transformer[Any, Any]):
    """ A composite transformer applies a list of transformers, one after each other, to the transformed object 
    and returns the result. """

    transformers: List[Transformer]
    data: Any

    def __init__(self, transformers = List[Transformer]):
        self.transformers = transformers

    def transform(self, data: Any) -> Any:
        self.data = data
        for transformer in self.transformers:
            try:
                self.data = transformer.transform(data = self.data)
            except Exception as exception:
                raise TransformError("error while applying composite transformer", exception)
        return self.data
