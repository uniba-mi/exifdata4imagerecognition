from tabnanny import verbose
from typing import List, TypeVar
from DataSource.TrainingDataSource import TrainingDataSource
from Transformers.Transformer import TransformError, Transformer

T = TypeVar('T')

class DataSourceInstanceTransformer(Transformer[TrainingDataSource[T], TrainingDataSource[T]]):
    """ Transforms the all instances of a data source with the given transformer. When a specific property of the data source's
    instances shall be transformed, the name of the property must be applied to the variable 'variableSelector' and 
    the type of the property must conform to the transforming type. Note: The transformer directly alters the
    state of the data source which can be undone by reloading the data source. """

    transformers: List[Transformer[T, T]]
    variableSelector: str
    errorCount: int
    verbose: int

    def __init__(self, transformers: List[Transformer[T, T]], variableSelector: str = None, verbose: int = 1):
        self.transformers = transformers
        self.variableSelector = variableSelector
        self.errorCount = 0
        self.verbose = verbose

    def transform(self, data: TrainingDataSource[T]) -> TrainingDataSource[T]:
        self.errorCount = 0
        for concept in data.getConcepts():
            concept.instances = { transformed for instance in concept.instances if (transformed := self.__transformInstance(instance)) is not None }
        return data

    def __transformInstance(self, instance: T) -> T:
        transformableObject = getattr(instance, self.variableSelector, None) if not self.variableSelector == None else instance

        for transformer in self.transformers:
            if not transformableObject == None:
                try:
                    transformableObject = transformer.transform(data = transformableObject)
                except TransformError as exception:
                    self.errorCount += 1
                    if self.verbose > 0:
                        print(exception)
                    return None
            else:
                break
        
        if not self.variableSelector == None and not transformableObject == None:
            setattr(instance, self.variableSelector, transformableObject)
            return instance

        return transformableObject

        
