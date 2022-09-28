from typing import Set, Tuple
from typing import TypeVar, Generic

T = TypeVar('T')

class TrainingConcept(Generic[T]):
    """ A "TrainingConcept" represents a training class (target concept). It provides training instances that can be used to train
    ML algorithms. A training concept may consist of sub-concepts. The instances of a top-level concept are given by
    the instances of all sub-level concepts plus the instances of the top-level concept. """

    names: Tuple
    superConceptNames: Tuple
    instances: Set[T]

    def __init__(self, names: Tuple, superConceptNames: Tuple = None):
        if not names == None and len(names) > 0:
            self.names = names
            self.superConceptNames = superConceptNames
            self.instances = set()
        else:
            raise ValueError("there must be at least one name for a training concept.")
       
    def addTrainingInstance(self, instance: T):
        """ adds a training instance to the concept. """
        self.instances.add(instance)
    
    def addTrainingInstances(self, instances: Set[T]):
        """ adds a set of training instances to the concept. """
        for instance in instances:
            self.addTrainingInstance(instance)

    def __str__(self):
        retVal =  "TrainingConcept - name = " + self.name
        if not self.superConceptName == None:
            retVal += ", superConceptName = " + self.superConceptName
        retVal += ", instanceCount = " + str(len(self.instances))
        return retVal
