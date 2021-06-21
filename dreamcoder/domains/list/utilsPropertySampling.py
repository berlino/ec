import dill
import numpy as np
import random

from dreamcoder.domains.list.property import Property
from dreamcoder.domains.list.utilsProperties import convertToPropertyTasks
from dreamcoder.domains.list.propSim import getPropertySimTasksMatrix
from dreamcoder.enumeration import multicoreEnumeration
from dreamcoder.likelihoodModel import TaskDiscriminationScore, UniqueTaskSignatureScore, TaskSurprisalScore

MIN_LOG_PRIOR = -11
DATA_DIR = "data/prop_sig/"
SAMPLED_PROPERTIES_DIR = "sampled_properties/"

def updateSavedPropertiesWithNewCacheTable(properties, propertiesPath):

    oldProperties = dill.load(open(propertiesPath, "rb"))

    for oldP in oldProperties:
        matchingProperty = [p for p in properties if p.name == oldP.name][0]
        for spec, propertyValue in matchingProperty.cachedTaskEvaluations.items():
            if spec in oldP.cachedTaskEvaluations:
                assert oldP.cachedTaskEvaluations[spec] == propertyValue
            else:
                oldP.cachedTaskEvaluations[spec] = propertyValue

    dill.dump(oldProperties, open(propertiesPath, "wb"))

    print("Updating cache table and rewriting properties at: {}".format(propertiesPath))
    return


def getPropertySamplingGrammar(baseGrammar, grammarName, frontiers, pseudoCounts=1, seed=0):
    if grammarName == "random":
        random.seed(seed)
        grammar = baseGrammar.randomWeights(r=lambda oldWeight: -1 * random.uniform(0,5))
    elif grammarName == "fitted":
        frontiersToFitOn = [f for f in frontiers if (len(f.entries) > 0 and baseGrammar.logLikelihood(f.task.request, f.topK(1).entries[0].program) > MIN_LOG_PRIOR)]
        print("Fitting on {} frontiers with logPrior > {}".format(len(frontiersToFitOn), MIN_LOG_PRIOR))
        grammar = baseGrammar.insideOutside(frontiersToFitOn, pseudoCounts=pseudoCounts)
    elif grammarName == "uniform":
        grammar = baseGrammar
    else:
        raise Exception("Provided sampling grammar weights argument: {} is invalid".format(grammarName))
    return grammar


def enumerateProperties(args, propertyGrammar, tasksToSolve, propertyRequest, allTasks=None):

    # if we sample properties by "unique_task_signature" we don't need to enumerate the same properties
    # for every task, as whether we choose to include the property or not depends on all tasks.
    if args["propScoringMethod"] == "unique_task_signature":
        likelihoodModel = UniqueTaskSignatureScore(timeout=0.1, tasks=allTasks)
    elif args["propScoringMethod"] == "general_unique_task_signature":
        likelihoodModel = GeneralUniqueTaskSignatureScore(timeout=0.1, tasks=allTasks)
    elif args["propScoringMethod"] == "per_task_surprisal":
        likelihoodModel = TaskSurprisalScore(timeout=0.1, tasks=allTasks)
    else:
        raise NotImplementedError

    print("Enumerating with {} CPUs".format(args["propCPUs"]))
    frontiers, times, pcs, likelihoodModel = multicoreEnumeration(propertyGrammar, tasksToSolve, solver=args["propSolver"],maximumFrontier= int(10e7),
                                                 enumerationTimeout= args["propSamplingTimeout"], CPUs=args["propCPUs"],
                                                 evaluationTimeout=0.01,
                                                 testing=True, likelihoodModel=likelihoodModel)

    if args["propScoringMethod"] == "general_unique_task_signature":
        print("{} properties of type: {}".format(len(likelihoodModel.properties), propertyRequest))
        for propertyInfo in likelihoodModel.properties:
            program, allSameValues = propertyInfo
            print("program: {}".format(program))
            print("allSameValues: {}".format(allSameValues))

        raise Exception("debug")

    propertiesPerTask = {}
    for frontier in frontiers:
        properties = []

        for entry in frontier.entries:
            prop = Property(program=entry.program.evaluate([]), 
                            request=propertyRequest, 
                            name=str(entry.program), 
                            logPrior=entry.logPrior, 
                            score=entry.logLikelihood)
            properties.append(prop)

        propertiesPerTask[frontier.task] = properties

    return propertiesPerTask, likelihoodModel


def propertyEnumerationMain(grammar, tasks, propertyRequest, featureExtractor, featureExtractorArgs):
    try:
        propertyFeatureExtractor = featureExtractor(tasks=tasks, similarTasks=None, grammar=grammar, cuda=False, featureExtractorArgs=featureExtractorArgs, propertyRequest=propertyRequest)
    # assertion triggered if 0 properties enumerated
    except AssertionError:
        print("0 properties found")
    # print("\nIteration {}: Found {} new properties".format(i, len(propertyFeatureExtractor.properties)))
    
    allProperties = propertyFeatureExtractor.properties
    print("--------------------------- Found {} properties -----------------------------".format(len(allProperties)))
    for prop in allProperties:
        print("Property: {} ({})".format(prop, prop.logPrior))

    propertyTasks = convertToPropertyTasks(tasks, propertyRequest=propertyRequest)
    scoreModel = TaskDiscriminationScore(timeout=0.1, tasks=propertyTasks)
    for task, propertyTask in zip(tasks, propertyTasks):
        tasksForPropertyScoring = [t for t in propertyTasks if t != propertyTask]
        tasksPropertyScores = [(scoreModel.scoreProperty(prop, propertyTask, tasksForPropertyScoring), prop) for prop in allProperties]
        print("---------------------------------------------------------------------------------\n{}\n".format(task.describe()))
        sortedTaskPropertyScores = sorted(tasksPropertyScores, reverse=True, key=lambda x: x[0][1])
        for evalRes, prop in sortedTaskPropertyScores[:10]:
            print("Score: {:.2f} - Value: {} - {}".format(evalRes[1], evalRes[0], prop))

    return allProperties


def filterProperties(properties, tasks, maxFractionSame, taskPropertyValueToInt, save=False, filename=None):

    properties = sorted(properties, key=lambda p: p.logPrior, reverse=True)

    propTasksMatrix = getPropertySimTasksMatrix(tasks, properties, taskPropertyValueToInt)
    print(propTasksMatrix.shape)

    def fractionSame(a, b):
        return np.sum((a == b).astype(int)) / a.shape[0]

    filteredTaskSigs = []
    filteredProperties = []
    for i in range(propTasksMatrix.shape[1]):
        taskSigToConsider = propTasksMatrix[:, i]
        if all([fractionSame(taskSigToConsider, taskSig) < maxFractionSame for taskSig in filteredTaskSigs]):
            filteredTaskSigs.append(taskSigToConsider)
            filteredProperties.append(properties[i])
            print("Passed: {}".format(properties[i]))
        else:
            print("Too similar: {}".format(properties[i]))

    print("Kept {} from {} properties".format(len(filteredProperties), len(properties)))
    if save:
        path = DATA_DIR + SAMPLED_PROPERTIES_DIR + fileName
        dill.dump(filteredProperties, open(path, "wb"))
        print("Saved filtered properties at: {}".format(path))
    return filteredProperties


# fileName = "sampled_properties_weights=fitted_sampling_timeout={}s_return_types=[bool]_seed=1.pkl".format(60)
# path = DATA_DIR + SAMPLED_PROPERTIES_DIR + fileName
# properties = dill.load(open(path, "rb"))
# properties = list(properties.values())[0] if len(properties) == 1 else properties
# maxFractionSame = 0.9
# fileName = "sampled_properties_weights=fitted_sampling_timeout=60s_return_types=[bool]_seed=1_filtered=True_maxFractionSame={}.pkl".format(maxFractionSame)
# taskPropertyValueToInt = {"allTrue":1, "mixed":2, "allFalse":0}
# filterProperties(properties, tasks, maxFractionSame, taskPropertyValueToInt, save=False, filename=fileName)