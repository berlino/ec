
from dreamcoder.domains.list.propSim import getPropSimGrammars
from dreamcoder.enumeration import multicoreEnumeration

class PropSimModel:
    def __init__(self,featureExtractor,grammar,
                 rank=None,contextual=False,mask=False,
                 cuda=False,
                 id=0):
        self.id = id
        self.fitted = False
        self.use_cuda = cuda

        self.featureExtractor = featureExtractor
        self.contextual = contextual
        self.grammar = ContextualGrammar.fromGrammar(grammar) if contextual else grammar
        self.generativeModel = grammar

        if cuda: self.cuda()

    def fit(self, 
        taskBatch, 
        allTasks, 
        helmholtzFrontiers, 
        onlyUseTrueProperties, 
        nSim, 
        propPseudocounts,
        weightedSim,
        weightByPrior,
        recomputeTasksWithTaskSpecificInputs,
        computePriorFromTasks,
        filterSimilarProperties,
        maxFractionSame,
        valuesToInt,
        verbose):

        task2FittedGrammar, tasksSolved, _ = getPropSimGrammars(
           self.grammar,
           taskBatch,
           allTasks, 
           helmholtzFrontiers, 
           self.featureExtractor.properties,
           onlyUseTrueProperties, 
           nSim, 
           propPseudocounts, 
           weightedSim, 
           compressSimilar=False, 
           weightByPrior=weightByPrior,
           recomputeTasksWithTaskSpecificInputs=recomputeTasksWithTaskSpecificInputs,
           computePriorFromTasks=computePriorFromTasks, 
           filterSimilarProperties=filterSimilarProperties, 
           maxFractionSame=maxFractionSame, 
           valuesToInt=valuesToInt,
           propSimIteration=0,
           verbose=verbose)

        self.task2FittedGrammar = task2FittedGrammar
        return self

    def enumerateFrontiers(self, taskBatch, CPUs, maximumFrontier, enumerationTimeout, evaluationTimeout, solver):
        return multicoreEnumeration(self.task2FittedGrammar, taskBatch, _=None,
                             enumerationTimeout=enumerationTimeout,
                             solver=solver,
                             CPUs=CPUs,
                             maximumFrontier=maximumFrontier,
                             verbose=True,
                             evaluationTimeout=evaluationTimeout,
                             testing=False,
                             likelihoodModel=None,
                             leaveHoldout=True)


