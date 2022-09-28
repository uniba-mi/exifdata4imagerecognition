from abc import ABC, abstractmethod
from typing import Callable, Generator, List, Set, Tuple
from DomainModel.TrainingConcept import TrainingConcept
from typing import TypeVar, Generic

T = TypeVar('T')

class TrainingDataSource(ABC, Generic[T]):
    """ a data source provides training data (concepts) for ML algorithms. """

    def getConcept(self, names: Tuple) -> TrainingConcept[T]:
        """ returns the training data for the concept of the given name tuple. """
        return next((concept for concept in self.getConcepts() if concept.names == names), None)
    
    def getConceptsForSuperConcept(self, names: Tuple) -> TrainingConcept[T]:
        """ returns the training data concepts which have a super concept with the given name tuple. """
        return [concept for concept in self.getConcepts() if concept.superConceptNames == names]

    def getSuperConceptNames(self) -> Set[Tuple]:
        """ returns a set of all super concept name tuples provided by this data source. """
        return {concept.superConceptNames for concept in self.getConcepts()}

    @abstractmethod
    def getConceptNames(self) -> Set[Tuple]:
        """ returns a set of all concept name tuples provided by this data source. """
        pass
    
    @abstractmethod
    def getConcepts(self) -> Set[TrainingConcept[T]]:
        """ returns the training data for all concepts. """
        pass
    
    @abstractmethod
    def reloadTrainingData(self):
        """ reloads the data of the data source. """
        pass
    
    @property
    def allInstances(self) -> Generator[T, None, None]:
        """ returns a generator for all instances of the data source. """
        for concept in self.getConcepts():
            for instance in concept.instances:
                yield instance


class CompositeTrainingDataSource(TrainingDataSource[T]):
    """ a composite data source combines multiple data sources. """

    dataSources: List[TrainingDataSource[T]]

    def __init__(self):
        self.dataSources = []

    def addDataSource(self, dataSource: TrainingDataSource[T]):
        if not dataSource in self.dataSources:
            self.dataSources.append(dataSource)
    
    def removeDataSource(self, dataSource: TrainingDataSource[T]):
        if dataSource in self.dataSources:
            self.dataSources.remove(dataSource)

    def getConceptNames(self) -> Set[Tuple]:
        names = set()
        for dataSource in self.dataSources:
            for conceptName in dataSource.getConceptNames():
                names.add(conceptName)
        return names

    def getConcept(self, names: Tuple) -> TrainingConcept[T]:
        concepts = []

        for dataSource in self.dataSources:
            concept = dataSource.getConcept(names = names)
            if not concept == None:
                if any(c.superConceptNames != concept.superConceptNames for c in concepts):
                    raise ValueError("composite data source contains multiple concepts with the same name but with different super concepts.")
                else:
                    concepts.append(concept)
                
        if len(concepts) > 0:
            compositeConcept = TrainingConcept[T](names = concepts[0].names, superConceptNames = concepts[0].superConceptNames)
            for concept in concepts:
                compositeConcept.addTrainingInstances(concept.instances)
            return compositeConcept
        
        return None

    def getConcepts(self) -> Set[TrainingConcept[T]]:
        concepts = set()

        for conceptNames in self.getConceptNames():
            concepts.add(self.getConcept(names = conceptNames))
        
        return concepts

    def reloadTrainingData(self):
        for dataSource in self.dataSources:
            dataSource.reloadTrainingData()

    def collect(self) -> "StaticTrainingDataSource[T]":
        """ collects all training concepts of all attached data sources and creates a new datasource which is 
        independent from its base sources. """
        return StaticTrainingDataSource(concepts = self.getConcepts)
    

class StaticTrainingDataSource(TrainingDataSource[T]):
    """ A static in-memory data source which is fixed upon creation time and cannot be reloaded. The training instances of the
    data source can be transformed and filtered. """

    concepts: Set[TrainingConcept[T]]
    
    def __init__(self, concepts: Set[TrainingConcept[T]]):
        names = [concept.names for concept in concepts]
        if len(names) == len(set(names)):
            self.concepts = concepts
        else:
            raise ValueError("a static data source may not contain multiple concepts with the same name", names)

    def getConceptNames(self) -> Set[Tuple]:
        return set([concept.names for concept in self.concepts])
    
    def getConcepts(self) -> Set[TrainingConcept[T]]:
        return self.concepts
    
    def reloadTrainingData(self):
        # cannot be done since the concepts of the data source are fixed
        raise RuntimeError("a static training data source cannot be reloaded.")

    def filter(self, filterFunction: Callable[[TrainingConcept], bool]):
        """ Removes all concepts from the data source for which the given function evaluates to true. """
        self.concepts = { concept for concept in self.concepts if not filterFunction(concept) }