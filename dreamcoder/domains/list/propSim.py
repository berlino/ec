
import numpy as np
import pandas as pd
from dreamcoder.utilities import vprint

def getPropertySimTasksMatrix(allFrontiers, properties, taskPropertyValueToInt):
    """

    Returns:
        A matrix of size (len(allFrontiers), len(properties))
    """
    matrix = []
    for f in allFrontiers:
        taskSig = [taskPropertyValueToInt[prop.getValue(f.task)] for prop in properties]
        matrix.append(taskSig)
    return np.array(matrix)


def getTaskSimilarFrontier(allFrontiers, propertyFeatureExtractor, propertySimTasksMatrix, valuesToInt, task, grammar, featureExtractorArgs, nSim=20, propertyToPriorDistribution=None, onlyUseTrueProperties=True, verbose=False):
    """
    Returns:
        frontiersToUse (list): A list of Frontier objects of the nSim most similar tasks to task. List of frontier is used to fit unigram / fit 
        bigram / train recognition model
    """

    vprint("\n------------------------------------------------ Task {} ----------------------------------------------------".format(task), verbose)
    vprint(task.describe(), verbose)
    simDf = createSimilarTasksDf(task, allFrontiers, propertyFeatureExtractor, propertySimTasksMatrix, propertyToPriorDistribution, valuesToInt, onlyUseTrueProperties=onlyUseTrueProperties, verbose=verbose)


    # check if any of the similar tasks programs are actually solutions for the task we want to solve
    solved = False
    for idx in simDf.head(nSim).index.values:
        simProgram = allFrontiers[idx].entries[0].program
        solved = task.check(simProgram, timeout=1)
        if solved:
            vprint("\nFound program solution: {}".format(simProgram), verbose)
            break


    vprint("\n{} Most Similar Prop Sig Tasks\n--------------------------------------------------------------------------------".format(nSim), verbose)
    frontiersToUse, frontierWeights = [], []
    for idx in simDf.head(nSim).index.values:        
        vprint("\nTask percent true property overlap: {}".format(simDf.loc[idx, "score"]), verbose)
        vprint(allFrontiers[idx].task.describe(), verbose)
        vprint("\nProgram ({}): {}".format(simDf.loc[idx, "programPrior"], simDf.loc[idx, "program"]), verbose)

        
        # evalReducedProgram = _reduceByEvaluating(simDf.loc[idx, "program"], {p.name: p for p in grammar.primitives})
        # vprint("\nEvaluation Reduced Program ({}): {}".format(grammar.logLikelihood(propertyFeatureExtractor.tasks[idx].request, evalReducedProgram), evalReducedProgram), verbose)

        # TODO: remove eventually
        frontier = allFrontiers[idx]
        frontiersToUse.append(allFrontiers[idx])
        frontierWeights.append(simDf.loc[idx, "score"])
    return frontiersToUse, frontierWeights, solved


def _getPriorBasedSimilarityScore(taskSig, trainTaskSig, propertyToPriorDistribution):

    taskProbsPerPropValue = propertyToPriorDistribution[taskSig, np.arange(propertyToPriorDistribution.shape[1])]
    trainTaskProbsPerPropValue = propertyToPriorDistribution[trainTaskSig, np.arange(propertyToPriorDistribution.shape[1])]   

    # The weight of each property p_i is the probability of (p_i(taskSig), p_i(trainTaskSig))
    weights = 1.0 / np.multiply(taskProbsPerPropValue,trainTaskProbsPerPropValue)
    
    # vector including only for which to tasks have the same value
    samePropertyVals = np.equal(trainTaskSig,taskSig)
    similarityVector = np.ones(taskSig.shape[0])
    similarityVector = np.multiply(similarityVector[samePropertyVals], weights[samePropertyVals])

    return np.sum(np.log(similarityVector))

def _getSimilarityScore(taskSig, trainTaskSig, onlyUseTrueProperties):

    similarityVector = np.zeros(taskSig.shape[0])
    samePropertyVals = np.equal(trainTaskSig,taskSig)

    if onlyUseTrueProperties:
        similarityVector[np.where((samePropertyVals) & (taskSig == 1))] = 1
        denominator = (taskSig == 1).sum()
    else:
        similarityVector[np.where((samePropertyVals))] = 1
        denominator = taskSig.shape[0]

    return np.sum(similarityVector.astype(int)) / denominator

def createSimilarTasksDf(task, allFrontiers, propertyFeatureExtractor, propertySimTasksMatrix, propertyToPriorDistribution, valuesToInt, onlyUseTrueProperties, verbose=False):
    """
    Args:
        task (Task): the task we want to solve.
        allFrontiers (list(Frontiers)): list of all the sampled frontiers
        propertyFeatureExtractor (PropertySignatureExtractor): the property signature feature extractor to use to compute properties
        propertySimTasksMatrix (np.ndarray): 2d numpy array of size (# of sampled frontiers, # of properties)
        propertyToPriorDistribution (np.ndarray): 2d numpy array of size (# unique property values, # properties)
        onlyUseTrueProperties (boolean): whether to calculate similarity score only using true properties of task or all of them. if 
        onlyUseTrueProperties any task with allTrue for all allTrue properties of task, will be deemed 100% similar even if for other
        properties their values differ.

    Returns:
        A dataframe where rows are tasks (indexed by task idx corresponding to propertyFeatureExtractor.tasks) with one column for the 
        similarity score of every task

    """

    taskSig = np.array([valuesToInt[prop.getValue(task)] for prop in propertyFeatureExtractor.properties])

    if onlyUseTrueProperties:
        propertiesMask = taskSig == valuesToInt["allTrue"]
    else:
        # only keep properties that aren't mixed for task we want to solve
        propertiesMask = np.logical_or(taskSig == valuesToInt["allTrue"], taskSig == valuesToInt["allFalse"])

    # filter properties only keeping ones with values desired for task to solve
    taskSig = taskSig[propertiesMask]
    propertyToPriorDistribution = propertyToPriorDistribution[:, propertiesMask]
    properties = [p for i,p in enumerate(propertyFeatureExtractor.properties) if propertiesMask[i]]
    propertySimTasksMatrix = propertySimTasksMatrix[:, propertiesMask]

    taskProbsPerPropValue = list(propertyToPriorDistribution[taskSig, np.arange(propertyToPriorDistribution.shape[1])])
    sortedTaskProbsPerPropValue = sorted([(i,score) for i,score in enumerate(taskProbsPerPropValue)], key=lambda x: x[1], reverse=False)

    n = min(20, len(properties))
    vprint("{} Highest scoring properties:".format(n), verbose)
    for i,propertyScore in sortedTaskProbsPerPropValue[:n]:
        vprint("{} -> {} ({})".format(propertyScore, properties[i], properties[i].getValue(task)), verbose)

    df = pd.DataFrame(data=propertySimTasksMatrix.T, index=[p.name for p in properties])
    if propertyToPriorDistribution is not None:
        simSeries = df.apply(lambda col: _getPriorBasedSimilarityScore(taskSig, col, propertyToPriorDistribution), axis=0)
    else:
        simSeries = df.apply(lambda col: _getSimilarityScore(taskSig, col, onlyUseTrueProperties), axis=0)
    simDf = simSeries.to_frame(name="score")

    # normalize score to make in range 0 to 1
    # simDf["score"] = simDf["score"] / simDf["score"].max()
    simDf["program"] = [allFrontiers[int(idx)].entries[0].program for idx in simDf.index]
    simDf["programPrior"] = [allFrontiers[int(idx)].entries[0].logPrior for idx in simDf.index]
    simDf["maxOutputListLength"] = [-max([len(o) for i,o in allFrontiers[int(idx)].task.examples]) for idx in simDf.index]
    simDf = simDf.drop_duplicates(subset=["program"])

    return simDf.sort_values(["score", "programPrior", "maxOutputListLength"], ascending=False)


def getPriorDistributionsOfProperties(properties, propertySimTasksMatrix, valuesToInt):
    """
    Calculates the prior distribution of each property from propertySimTasksMatrix.

    Args:
        properties (list(Property)): List of properties
        propertySimTasksMatrix (np.ndarray): 2d numpy array of size (len(allFrontiers), len(properties))

    Returns:

    """

    def getDistributionOfVector(vector, uniqueValues, pseudoCounts=1):

        # each element in counts corresponds to the frequency count of its idx
        counts = np.bincount(vector, minlength=len(uniqueValues))
        counts = counts + pseudoCounts
        normalizedCounts = counts / np.sum(counts)
        return normalizedCounts

    # matrix of size (number of unique values, properties) with probalities of property values
    uniqueValues = set(value for value in valuesToInt.values())
    probs = np.zeros((len(uniqueValues), propertySimTasksMatrix.shape[1]))
    for i in range(propertySimTasksMatrix.shape[1]):
        probs[:, i] = getDistributionOfVector(propertySimTasksMatrix[:, i], uniqueValues)

    return probs


def getPropSimGrammars(baseGrammar, tasks, sampledFrontiers, propertyFeatureExtractor, featureExtractorArgs, onlyUseTrueProperties, nSimList, pseudoCounts, weightedSim, compressSimilar, weightByPrior, recomputeTasksWithTaskSpecificInputs, verbose):
    """
    Returns:
        grammars (dict): every key is a task (Task) and its value is its corresponding grammar (Grammar) fitted on the most similar tasks
        taskSolved (set): set of tasks solved running PropSim (i.e. for which one of the 10,000 enumerated programs satisfied all I/O examples)
        task2Frontiers (dict): every key is a task (Task) and its value is the corresponding frontiers (list(Frontier)), one frontier for each similar task
    """

    tasksSolved = set()
    grammars = {}
    for nSim in nSimList:
        grammars[nSim] = {}
    task2Frontiers = {}

    print("Creating Similar Task Matrix")
    # convert string values to integers for efficiency reasons
    valuesToInt = {"allFalse":0, "allTrue":1, "mixed":2}
    propertySimTasksMatrix = getPropertySimTasksMatrix(sampledFrontiers, propertyFeatureExtractor.properties, valuesToInt)
    print("Finished Creating Similar Task Matrix")
    propertyToPriorDistribution = getPriorDistributionsOfProperties(propertyFeatureExtractor.properties, propertySimTasksMatrix, valuesToInt)
    print("propertyToPriorDistribution", propertyToPriorDistribution.shape)
    for task in tasks:
        # use the sampled programs to create new specs with the same inputs as the task we want to solve
        print("Find {} most similar tasks for task: {}".format(nSimList[0], task.name))
        if recomputeTasksWithTaskSpecificInputs: 
            newFrontiers = createFrontiersWithInputsFromTask(sampledFrontiers, task)
        else:
            newFrontiers = sampledFrontiers
        frontiers, weights, solved = getTaskSimilarFrontier(newFrontiers, propertyFeatureExtractor, propertySimTasksMatrix, valuesToInt, task, baseGrammar, featureExtractorArgs, nSim=max(nSimList), propertyToPriorDistribution=propertyToPriorDistribution, onlyUseTrueProperties=onlyUseTrueProperties, verbose=verbose)
        task2Frontiers[task] = frontiers

        if compressSimilar:
            if len([f for f in frontiers if not f.empty]) == 0:
                eprint("No compression frontiers; not inducing a grammar this iteration.")
            else:
                raise NotImplementedError
                # taskGrammar, compressionFrontiers = induceGrammar(baseGrammar, frontiers,
                #                                               topK=1,
                #                                               pseudoCounts=pseudoCounts, a=0,
                #                                               aic=1.0, structurePenalty=1.5,
                #                                               topk_use_only_likelihood=False,
                #                                               backend="ocaml", CPUs=1, iteration=0)
                # print("Finished inducing grammar")
        else:
            for nSim in nSimList:
                weights = weights if weightedSim else None
                taskGrammar = baseGrammar.insideOutside(frontiers[:nSim], pseudoCounts, iterations=1, frontierWeights=weights, weightByPrior=weightByPrior)
                grammars[nSim][task] = taskGrammar
        if solved:
            tasksSolved.add(task)

    return grammars, tasksSolved, task2Frontiers
