
import numpy as np
import pandas as pd
from dreamcoder.grammar import Grammar
from dreamcoder.utilities import vprint

from dreamcoder.domains.list.utilsProperties import createFrontiersWithInputsFromTask

MAX_NUM_SIM_TASKS_TO_PRINT = 5

def getPropertySimTasksMatrix(helmholtzTasks, properties, taskPropertyValueToInt):
    """

    Returns:
        A matrix of size (len(allFrontiers), len(properties))
    """
    matrix = []
    for i,task in enumerate(helmholtzTasks):
        taskSig = [taskPropertyValueToInt[prop.getValue(task)] for prop in properties]
        matrix.append(taskSig)
    return np.array(matrix)


def getTaskSimilarFrontier(
    allFrontiers, 
    properties, 
    propertySimTasksMatrix, 
    valuesToInt, 
    allTasks,
    taskIdx, 
    grammar,
    filterSimilarProperties, 
    maxFractionSame, 
    nSim=20, 
    propertyToPriorDistribution=None, 
    onlyUseTrueProperties=True, 
    recomputeTasksWithTaskSpecificInputs=False,
    computePriorFromTasks=False,
    verbose=False):
    """
    Returns:
        frontiersToUse (list): A list of Frontier objects of the nSim most similar tasks to task. List of frontier is used to fit unigram / fit 
        bigram / train recognition model
    """
    task = allTasks[taskIdx]
    vprint("\n------------------------------------------------ Task {} ----------------------------------------------------".format(task), verbose)
    vprint(task.describe(), verbose)
    simDf, matchingFrontiers = createSimilarTasksDf(allTasks, taskIdx, allFrontiers, properties, propertySimTasksMatrix, propertyToPriorDistribution, valuesToInt, 
        onlyUseTrueProperties=onlyUseTrueProperties, filterSimilarProperties=filterSimilarProperties, maxFractionSame=maxFractionSame, 
        recomputeTasksWithTaskSpecificInputs=recomputeTasksWithTaskSpecificInputs, computePriorFromTasks=computePriorFromTasks, verbose=verbose)


    # check if any of the similar tasks programs are actually solutions for the task we want to solve
    solved = False
    for idx in simDf.head(nSim).index.values:
        simProgram = matchingFrontiers[idx].entries[0].program
        solved = task.check(simProgram, timeout=1)
        if solved:
            print("\nFound program solution for task {}: {}".format(task, simProgram))
            break


    vprint("\n{} Most Similar Prop Sig Tasks\n--------------------------------------------------------------------------------".format(nSim), verbose)
    frontiersToUse, frontierWeights = [], []
    for i,idx in enumerate(simDf.head(nSim).index.values):
        if i < MAX_NUM_SIM_TASKS_TO_PRINT:
            vprint("\nTask similarity score: {}".format(simDf.loc[idx, "score"]), verbose)
            vprint(matchingFrontiers[idx].task.describe(), verbose)
            vprint("\nProgram ({}): {}".format(simDf.loc[idx, "programPrior"], simDf.loc[idx, "program"]), verbose)
        
        # evalReducedProgram = _reduceByEvaluating(simDf.loc[idx, "program"], {p.name: p for p in grammar.primitives})
        # vprint("\nEvaluation Reduced Program ({}): {}".format(grammar.logLikelihood(propertyFeatureExtractor.tasks[idx].request, evalReducedProgram), evalReducedProgram), verbose)

        frontiersToUse.append(matchingFrontiers[idx])
        frontierWeights.append(simDf.loc[idx, "score"])

    return frontiersToUse, frontierWeights, solved


def _getPriorBasedSimilarityScore(taskSig, trainTaskSig, propertyToPriorDistribution):

    taskProbsPerPropValue = propertyToPriorDistribution[taskSig, np.arange(propertyToPriorDistribution.shape[1])]
    trainTaskProbsPerPropValue = propertyToPriorDistribution[trainTaskSig, np.arange(propertyToPriorDistribution.shape[1])]   

    # The weight of each property p_i is the probability of (p_i(taskSig), p_i(trainTaskSig))
    weights = 1.0 / np.multiply(taskProbsPerPropValue,trainTaskProbsPerPropValue)
    
    # vector including only properties for which the 2 tasks have the same value
    samePropertyVals = np.equal(trainTaskSig,taskSig)
    similarityVector = np.ones(taskSig.shape[0])
    similarityVector = np.multiply(similarityVector[samePropertyVals], weights[samePropertyVals])

    # return np.sum(similarityVector)
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


def filterProperties(properties, propTasksMatrix, maxFractionSame, save=False, filename=None):

    def fractionSame(a, b):
        return np.sum((a == b).astype(int)) / a.shape[0]

    assert len(properties) == propTasksMatrix.shape[1]

    filteredPropertyIds = []
    for i,p in enumerate(properties):
        taskSigToConsider = propTasksMatrix[:, i]
        if all([fractionSame(taskSigToConsider, propTasksMatrix[:, j]) < maxFractionSame for j in filteredPropertyIds]):
            filteredPropertyIds.append(i)

    filteredProperties = [properties[j] for j in filteredPropertyIds]
    print("Kept {} from {} properties".format(len(filteredProperties), propTasksMatrix.shape[1]))
    if save:
        path = DATA_DIR + SAMPLED_PROPERTIES_DIR + fileName
        dill.dump(filteredProperties, open(path, "wb"))
        print("Saved filtered properties at: {}".format(path))
    return filteredProperties


def createSimilarTasksDf(
    allTasks, 
    taskIdx, 
    allFrontiers, 
    properties, 
    propertySimTasksMatrix, 
    propertyToPriorDistribution, 
    valuesToInt, 
    onlyUseTrueProperties, 
    filterSimilarProperties, 
    maxFractionSame, 
    recomputeTasksWithTaskSpecificInputs, 
    computePriorFromTasks,
    verbose=False):
    """
    Args:
        task (Task): the task we want to solve.
        allFrontiers (list(Frontiers)): list of all the sampled frontiers
        properties (list(Property)): the list of properties to use for PropSim score
        propertySimTasksMatrix (np.ndarray): 2d numpy array of size (# of sampled frontiers, # of properties)
        propertyToPriorDistribution (np.ndarray): 2d numpy array of size (# unique property values, # properties)
        onlyUseTrueProperties (boolean): whether to calculate similarity score only using true properties of task or all of them. if 
        onlyUseTrueProperties any task with allTrue for all allTrue properties of task, will be deemed 100% similar even if for other
        properties their values differ.

    Returns:
        A dataframe where rows are tasks (indexed by task idx corresponding to propertyFeatureExtractor.tasks) with one column for the 
        similarity score of every task

    """
    task = allTasks[taskIdx]
    taskSig = np.array([valuesToInt[prop.getValue(task)] for prop in properties])
    if onlyUseTrueProperties:
        propertiesMask = (taskSig == valuesToInt["allTrue"])
    else:
        # only keep properties that aren't mixed for task we want to solve
        propertiesMask = np.logical_or(taskSig == valuesToInt["allTrue"], taskSig == valuesToInt["allFalse"])

    # filter properties only keeping ones with values desired for task to solve
    propertyToIdx = {p:i for i,p in enumerate(properties)}
    properties = [p for i,p in enumerate(properties) if propertiesMask[i]]
    
    # this will only be true on the first iteration of iterative propSim where we use the same frontier for alll tasks
    if propertySimTasksMatrix is not None and propertyToPriorDistribution is not None:
        pass
    else:
        # we use the sampled programs but execute on inputs of tasks we want to solve
        if recomputeTasksWithTaskSpecificInputs: 
            allFrontiers = createFrontiersWithInputsFromTask(allFrontiers, task)

        propertySimTasksMatrix, propertyToPriorDistribution = _getSimTaskMatrixAndPropertyPriors(allTasks, allFrontiers, properties, valuesToInt, computePriorFromTasks)

        # update propertyToIdx to point to the idx of property in the new data structures
        propertyToIdx = {p:i for i,p in enumerate(properties)}
        vprint("new shape of propertyToPriorDistribution: {}".format(propertyToPriorDistribution.shape), verbose)
        vprint("new shape of propertySimTasksMatrix: {}".format(propertySimTasksMatrix.shape), verbose)

    # sorted properties by (1 / prior probability) of observed value
    propertyScores = list(propertyToPriorDistribution[taskSig[propertiesMask], [propertyToIdx[prop] for prop in properties]])
    sortedPropAndScores = sorted([(prop, score) for prop,score in zip(properties, propertyScores)], key=lambda x: x[1], reverse=False)
    properties = [el[0] for el in sortedPropAndScores]

    if filterSimilarProperties:
        # filter properties only keeping high-scoring "independent" ones
        idxs = [propertyToIdx[prop] for prop in properties]
        reorderedPropertySimTasksMatrix = propertySimTasksMatrix[:, idxs].copy()
        reorderedPropertySimTasksMatrix[reorderedPropertySimTasksMatrix != valuesToInt["allTrue"]] = 0
        properties = filterProperties(properties=properties, propTasksMatrix=reorderedPropertySimTasksMatrix, maxFractionSame=maxFractionSame, save=False, filename=None)
        sortedPropAndScores = [(p,score) for p,score in sortedPropAndScores if p in properties]

    taskSig = np.array([valuesToInt[prop.getValue(task)] for prop in properties])

    n = min(20, len(properties))
    vprint("{} Highest scoring properties:".format(n), verbose)
    for prop,score in sortedPropAndScores[:n]:
        vprint("{} -> {} ({})".format(score, prop, prop.getValue(task)), verbose)
        # vprint(propertyToPriorDistribution[:, propertyToIdx[prop]], verbose)
        # print(propertySimTasksMatrix[:, propertyToIdx[prop]])

    data = propertySimTasksMatrix[:, [propertyToIdx[p] for p in properties]]
    df = pd.DataFrame(data=data)
    if propertyToPriorDistribution is not None:
        simSeries = df.apply(lambda row: _getPriorBasedSimilarityScore(taskSig, row, propertyToPriorDistribution[:, [propertyToIdx[p] for p in properties]]), axis=1)
    else:
        simSeries = df.apply(lambda row: _getSimilarityScore(taskSig, row, onlyUseTrueProperties), axis=1)
    simDf = simSeries.to_frame(name="score")

    # normalize score to make in range 0 to 1
    # simDf["score"] = simDf["score"] / simDf["score"].max()
    simDf["program"] = [allFrontiers[int(idx)].entries[0].program for idx in simDf.index]
    simDf["programPrior"] = [allFrontiers[int(idx)].entries[0].logPrior for idx in simDf.index]
    simDf["maxOutputListLength"] = [-max([len(o) for i,o in allFrontiers[int(idx)].task.examples]) for idx in simDf.index]
    simDf = simDf.drop_duplicates(subset=["program"])
    simDf = simDf.sort_values(["score", "programPrior", "maxOutputListLength"], ascending=False)

    # assert that the simDf row index is aligned with the index of the corresponding frontier
    for i,frontier in enumerate(allFrontiers):
        assert simDf.loc[i, "program"] == frontier.entries[0].program

    return simDf, allFrontiers


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


def _getSimTaskMatrixAndPropertyPriors(allTasks, frontiers, properties, valuesToInt, computePriorFromTasks):

    print("Creating Similar Task Matrix")
    propertySimTasksMatrix = getPropertySimTasksMatrix([f.task for f in frontiers], properties, valuesToInt)
    print("Finished Creating Similar Task Matrix with size: {}".format(propertySimTasksMatrix.shape))
    if computePriorFromTasks:
        propertyValsMatrix = getPropertySimTasksMatrix(allTasks, properties, valuesToInt)
        propertyToPriorDistribution = getPriorDistributionsOfProperties(properties, propertyValsMatrix, valuesToInt)
    else:
        propertyToPriorDistribution = getPriorDistributionsOfProperties(properties, propertySimTasksMatrix, valuesToInt)
        print("propertyToPriorDistribution", propertyToPriorDistribution.shape)
    return propertySimTasksMatrix, propertyToPriorDistribution

def getPropSimGrammars(
    baseGrammars, 
    tasksToSolve,
    allTasks,
    sampledFrontiers, 
    properties,
    onlyUseTrueProperties, 
    nSim, 
    pseudoCounts, 
    weightedSim, 
    compressSimilar, 
    weightByPrior, 
    recomputeTasksWithTaskSpecificInputs, 
    computePriorFromTasks, 
    filterSimilarProperties, 
    maxFractionSame, 
    valuesToInt,
    propSimIteration,
    verbose):

    """
    Returns:
        task2FittedGrammar (dict): every key is a task (Task) and its value is its corresponding grammar (Grammar) fitted on the most similar tasks
        tasksSolved (set): set of tasks solved running PropSim (i.e. for which one of the 10,000 enumerated programs satisfied all I/O examples)
        task2SimilarFrontiers (dict): every key is a task (Task) and its value is the corresponding frontiers (list(Frontier)), one frontier for each similar task
    """

    task2SimilarFrontiers, task2FittedGrammar, tasksSolved = {}, {}, set()

    if not isinstance(baseGrammars, dict):
        baseGrammars = {t: baseGrammars for t in tasksToSolve}
    task2Grammar = baseGrammars

    if not isinstance(sampledFrontiers, dict):
        sampledFrontiers = {t: sampledFrontiers for t in tasksToSolve}
    task2Frontiers = sampledFrontiers

    if not isinstance(properties, dict):
        properties = {t: properties for t in tasksToSolve}
    task2Properties = properties

    propertySimTasksMatrix, propertyToPriorDistribution = None, None
    # we can get away with computing once for all tasks if the below conditions are True
    if not recomputeTasksWithTaskSpecificInputs and computePriorFromTasks and propSimIteration == 0:
        propertySimTasksMatrix, propertyToPriorDistribution = _getSimTaskMatrixAndPropertyPriors(allTasks, sampledFrontiers[tasksToSolve[0]], properties, valuesToInt, True)

    for taskIdx,task in enumerate(allTasks):
        # use the sampled programs to create new specs with the same inputs as the task we want to solve
        if task in tasksToSolve:
            similarFrontiers, weights, solved = getTaskSimilarFrontier(task2Frontiers[task], task2Properties[task], propertySimTasksMatrix, valuesToInt, allTasks, taskIdx, task2Grammar[task], 
                filterSimilarProperties=filterSimilarProperties, maxFractionSame=maxFractionSame, nSim=nSim, propertyToPriorDistribution=propertyToPriorDistribution, 
                onlyUseTrueProperties=onlyUseTrueProperties, recomputeTasksWithTaskSpecificInputs=recomputeTasksWithTaskSpecificInputs, computePriorFromTasks=computePriorFromTasks, verbose=verbose)
            task2SimilarFrontiers[task] = similarFrontiers
        else:
            continue

        if compressSimilar:
            if len([f for f in similarFrontiers if not f.empty]) == 0:
                eprint("No compression frontiers; not inducing a grammar this iteration.")
            else:
                raise NotImplementedError
        else:
            weights = weights if weightedSim else None
            vprint(similarFrontiers[:nSim][0].task.describe(), verbose)
            taskGrammar = task2Grammar[task].insideOutside(similarFrontiers[:nSim], pseudoCounts, iterations=1, frontierWeights=weights, weightByPrior=weightByPrior)
            vprint("\nGrammar after fitting for task {}:\n{}".format(task, taskGrammar), verbose)
            task2FittedGrammar[task] = taskGrammar
        if solved:
            tasksSolved.add(task)

    return task2FittedGrammar, tasksSolved, task2SimilarFrontiers


def enumerationProxy(task2FittedGrammar, train, frontiers, grammar, nSim, verbose=False):
    """
    Given a frontier of tasks prints out the logposterior of tasks in train using:
        - the RNN-encoded neural recogntion model
        - the unigram grammar fitted on nSim most similar tasks
    """

    uniformGrammarPriors, logVariableGrammarPriors, fittedLogPosteriors, baselineLogPosteriors = 0.0, 0.0, 0.0, 0.0
    taskToFrontier = {f.task:f for f in frontiers if len(f.entries) > 0}

    numTasks = 0
    fittedTasks = None
    if isinstance(task2FittedGrammar, list):
        fittedTasks = list(task2FittedGrammar[0].keys())
    else:
        fittedTasks = list(task2FittedGrammar.keys())
    for task in train:
        # if we have a fitted grammar for this task and a ground truth program we can score the grammar on
        if task in taskToFrontier and (task in fittedTasks):

            bestFrontier = taskToFrontier[task].topK(1)
            program = bestFrontier.entries[0].program
            numTasks += 1

            vprint("\n-------------------------------------------------------------------------------", verbose)
            vprint(task.describe(), verbose)
            vprint("Ground Truth Program: {}".format(program), verbose)
            vprint("---------------------------------------------------------------------------------", verbose)
            uniformGrammarPrior = grammar.logLikelihood(task.request, program)
            vprint("Uniform Grammar Prior: {}".format(uniformGrammarPrior), verbose)
            logVariableGrammar = Grammar(2.0, [(0.0, p.infer(), p) for p in grammar.primitives], continuationType=None)
            logVariableGrammarPrior = logVariableGrammar.logLikelihood(task.request, program)
            vprint("Log Variable Program Prior: {}".format(logVariableGrammarPrior), verbose)
            vprint("---------------------------------------------------------------------------------", verbose)

            if isinstance(task2FittedGrammar, list):
                toPrint = "PropSim Grammar LP ({} frontiers):".format(nSim)
                for task2Grammar in task2FittedGrammar:
                    if task in task2Grammar:     
                        fittedLogPosterior = task2Grammar[task].logLikelihood(task.request, program)
                        toPrint += " {} ->".format(fittedLogPosterior)
                    else:
                        toPrint += "solved"
                vprint(toPrint, verbose)
            else:
                if task in task2FittedGrammar:
                    fittedLogPosterior = task2FittedGrammar[task].logLikelihood(task.request, program)
                    vprint("ProSim Grammar LP ({} frontier): {}".format(nSim, fittedLogPosterior), verbose) 
            baselineLogPosterior = bestFrontier.entries[0].logPosterior
            vprint("Baseline LogPosterior: {}\n".format(baselineLogPosterior), verbose)
            
            
            uniformGrammarPriors += uniformGrammarPrior
            logVariableGrammarPriors += logVariableGrammarPrior
            baselineLogPosteriors += baselineLogPosterior
            fittedLogPosteriors += fittedLogPosterior

    if numTasks == 0:
        print("No solved frontiers from which to report metrics")
        return
    else:
        print("Metrics for {} tasks with solutions".format(numTasks))

    print("Mean Uniform Grammar Prior: {}".format(uniformGrammarPriors / numTasks))
    print("Mean Log Variable Grammar Prior: {}".format(logVariableGrammarPriors / numTasks))
    print("Mean Baseline Log Posterior: {}".format(baselineLogPosteriors / numTasks))
    print("Mean Fitted Log Posterior ({} frontiers): {}".format(nSim, fittedLogPosteriors / numTasks))

    return task2FittedGrammar


