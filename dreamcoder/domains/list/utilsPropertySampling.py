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
    elif grammarName == "same":
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


######################
# Sample Properties
######################

# fileName = "sampled_properties_weights=fitted_sampling_timeout={}s_return_types=[bool]_seed=1.pkl".format(60)
# path = DATA_DIR + SAMPLED_PROPERTIES_DIR + fileName
# properties = dill.load(open(path, "rb"))
# properties = list(properties.values())[0] if len(properties) == 1 else properties
# maxFractionSame = 0.9
# fileName = "sampled_properties_weights=fitted_sampling_timeout=60s_return_types=[bool]_seed=1_filtered=True_maxFractionSame={}.pkl".format(maxFractionSame)
# taskPropertyValueToInt = {"allTrue":1, "mixed":2, "allFalse":0}


######################
# Smarter PropSim
######################

# fileName = "enumerationResults/neuralRecognizer_2021-05-18 15:27:58.504808_t=600.pkl"
# frontiers, times = dill.load(open(fileName, "rb"))
# unsolvedTasks = [f.task for f in frontiers if len(f.entries) == 0]

# specificTasks = ["023_1"]
# if len(specificTasks) > 0:
#     unsolvedTasks = [t for t in unsolvedTasks if t.name in specificTasks]

# if propToUse == "handwritten":
#     properties = getHandwrittenPropertiesFromTemplates(train)

# featureExtractor = extractor([f.task for f in sampledFrontiers], grammar=baseGrammar, cuda=False, featureExtractorArgs=featureExtractorArgs, properties=properties)
# onlySampleFor100percentSimTasks = True
# for i,task in enumerate(train):
#     similarTaskFrontiers, frontierWeights, solved = getTaskSimilarFrontier(sampledFrontiers, featureExtractor, task, baseGrammar, featureExtractorArgs, nSim=5, onlyUseTrueProperties=True, verbose=True)
#     break

    # if onlySampleFor100percentSimTasks:
    #     print(len(similarTaskFrontiers))
    #     print(frontierWeights)
    #     hundredPercentSimilarTaskFrontiers = [f for j,f in enumerate(similarTaskFrontiers) if frontierWeights[j] == 1.0]
    #     if len(hundredPercentSimilarTaskFrontiers) == 0:
    #         continue
    #     else:
    #         print("Sampling properties to improve {} tasks with 100 percent true property overlap".format(len(hundredPercentSimilarTaskFrontiers)))
    #         newPropertyFrontiers = sampleProperties(featureExtractorArgs, baseGrammar, tasks=[task], 
    #             similarTasks=[f.task for f in hundredPercentSimilarTaskFrontiers], propertyRequest=arrow(tinput, toutput, tbool))
    #         print(newPropertyFrontiers)
    # else:
    #     newPropertyFrontiers = sampleProperties(featureExtractorArgs, propSamplingPrimitives, tasks=[task], 
    #         similarTasks=[f.task for f in similarTaskFrontiers], propertyRequest=arrow(tinput, toutput, tbool))
    #     print(newPropertyFrontiers)

# prims = {p.name:p for p in baseGrammar.primitives}
# programString = "(lambda (cut_slice 5 7 (take (second (take (* 3 9) (append $0 4))) (append (foldi (foldi $0 $0 (lambda (lambda (lambda empty)))) (slice 0 5 empty) (lambda (lambda (lambda (mapi (lambda (lambda 0)) $0))))) (if (is-even 2) (index 4 $0) 0)))))"
# p = Program.parse(programString)
# print("\nOriginal Program: {}\n".format(p))

# evalReduced, p = p.evalReduce(prims)
# while evalReduced:
#     evalReduced, temp = p.evalReduce(prims)
#     if temp == p:
#         break
#     else:
#         p = temp
# print("\nProgram after evalReductions: {}".format(p))