import copy
import datetime
import dill
import numpy as np
import pandas as pd
from threading import Thread,Lock

from dreamcoder.compression import induceGrammar
from dreamcoder.enumeration import multicoreEnumeration
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.grammar import Grammar
from dreamcoder.domains.list.propertySignatureExtractor import PropertySignatureExtractor
from dreamcoder.type import Context, arrow, tlist, tint
from dreamcoder.utilities import *


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

def _reduceByEvaluating(p, primitives):
    evalReduced, p = p.evalReduce(primitives)
    while evalReduced:
        evalReduced, temp = p.evalReduce(primitives)
        if temp == p:
            break
        else:
            p = temp
    return p


def makeTaskFromProgram(program, request, featureExtractor, differentOutputs=True, filterIdentityTask=True):
    task = featureExtractor.taskOfProgram(program, request)
    if task is None:
        return None
    else:
        if differentOutputs:
            if all([o == task.examples[0][1] for i,o in task.examples]):
                return None
        if filterIdentityTask:
            if all([i[0] == o for i,o in task.examples]):
                return None
    return task

def enumerateAndSave(grammar, request, featureExtractor, dslName, numTasks, k, batchSize, CPUs=1):

    def enumerateWithinBounds(lowerBound, upperBound, totalNumTasks):

        totalNumTasksEnumerated = 0.0
        enumeratedFrontiersBatch = []
        for logPrior, context, p in grammar.enumeration(Context.EMPTY, [], request, upperBound, maximumDepth=99, lowerBound=lowerBound):
            task = makeTaskFromProgram(p, request, featureExtractor, differentOutputs=True, filterIdentityTask=True)
            if task is not None:
                frontier = Frontier([FrontierEntry(program=p,
                                                   logLikelihood=0., logPrior=logPrior)],
                                    task=task)
                enumeratedFrontiersBatch.append(frontier)
                totalNumTasksEnumerated += 1
                print(totalNumTasksEnumerated, logPrior)

        if totalNumTasksEnumerated > 0:
            writePath = "data/prop_sig/{}_enumerated_{}/enumerated_{}_{}.pkl".format(dslName, k, lowerBound, upperBound)
            dill.dump(enumeratedFrontiersBatch, open(writePath, "wb"))
            print("Writing tasks to: {}".format(writePath))
        totalNumTasks[(lowerBound, upperBound)] = totalNumTasksEnumerated
        return

    bounds = [((lowerBound/2.0), (lowerBound/2.0) + 0.5) for lowerBound in range(20,40)]
    totalNumTasks = {bound: 0 for bound in bounds}

    if CPUs > 1:
        parallelMap(CPUs, lambda bounds: enumerateWithinBounds(bounds[0], bounds[1], totalNumTasks), bounds)

    print(totalNumTasks)
    return


def getTaskSimilarFrontier(allFrontiers, propertyFeatureExtractor, task, grammar, featureExtractorArgs, nSim=20, onlyUseTrueProperties=True, verbose=False):
    """
    Returns:
        frontiersToUse (list): A list of Frontier objects of the nSim most similar tasks to task. List of frontier is used to fit unigram / fit 
        bigram / train recognition model
    """

    vprint("\n------------------------------------------------ Task {} ----------------------------------------------------".format(task), verbose)
    vprint(task.describe(), verbose)
    simDf = getSimilarTasksByProperty(task, allFrontiers, propertyFeatureExtractor, onlyUseTrueProperties=onlyUseTrueProperties, verbose=verbose)


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
        vprint(propertyFeatureExtractor.tasks[idx].describe(), verbose)
        vprint("\nProgram ({}): {}".format(simDf.loc[idx, "programPrior"], simDf.loc[idx, "program"]), verbose)
        evalReducedProgram = _reduceByEvaluating(simDf.loc[idx, "program"], {p.name: p for p in grammar.primitives})
        vprint("\nEvaluation Reduced Program ({}): {}".format(grammar.logLikelihood(propertyFeatureExtractor.tasks[idx].request, evalReducedProgram), evalReducedProgram), verbose)

        # TODO: remove eventually
        frontier = allFrontiers[idx]
        frontiersToUse.append(allFrontiers[idx])
        frontierWeights.append(simDf.loc[idx, "score"])
    return frontiersToUse, frontierWeights, solved


def getSimilarTasksByProperty(task, allFrontiers, propertyFeatureExtractor, onlyUseTrueProperties, verbose=False):
    """
    Args:
        task (Task): the task we want to solve.
        allFrontiers (list(Frontiers)): list of all the sampled frontiers
        propertyFeatureExtractor (PropertySignatureExtractor): the property signature feature extractor to use to compute properties
        onlyUseTrueProperties (boolean): whether to calculate similarity score only using true properties of task or all of them. if 
        onlyUseTrueProperties any task with allTrue for all allTrue properties of task, will be deemed 100% similar even if for other
        properties their values differ.

    Returns:
        A dataframe where rows are tasks (indexed by task idx corresponding to propertyFeatureExtractor.tasks) with one column for the 
        similarity score of every task

    """

    assert isinstance(propertyFeatureExtractor, PropertySignatureExtractor)

    propertyFeatureExtractor.featuresOfTask(task)
    taskSig = propertyFeatureExtractor.booleanPropSig.numpy()
    truePropertiesIdx = list(np.argwhere(taskSig == 1).squeeze())
    vprint("\n{} true properties for task {}\n".format(len(truePropertiesIdx), task.name), verbose)
    for idx in truePropertiesIdx:
        vprint(propertyFeatureExtractor.properties[idx][0], verbose)

    # for performance reasons we can only keep the columns (properties) that are true for the task of interest
    if onlyUseTrueProperties:
        properties = [el for i,el in enumerate(propertyFeatureExtractor.properties) if i in truePropertiesIdx]
        taskSig = taskSig[truePropertiesIdx]
    else:
        properties = propertyFeatureExtractor.properties

    data, propertyNames = [], []
    for (propertyName, f, propertySig) in properties:
        data.append(propertySig)
        propertyNames.append(propertyName)

    df = pd.DataFrame(data=data, index=propertyNames)

    simSeries = df.apply(lambda col: _getSimilarityScore(taskSig, col, onlyUseTrueProperties), axis=0)
    simDf = simSeries.to_frame(name="score")

    simDf["program"] = [allFrontiers[int(idx)].entries[0].program for idx in simDf.index]
    simDf["programPrior"] = [allFrontiers[int(idx)].entries[0].logPrior for idx in simDf.index]
    simDf["maxOutputListLength"] = [-max([len(o) for i,o in allFrontiers[int(idx)].task.examples]) for idx in simDf.index]
    simDf = simDf.drop_duplicates(subset=["program"])

    return simDf.sort_values(["score", "programPrior", "maxOutputListLength"], ascending=False)


def sampleAndSave(recognitionModel, requests, dslName, numSamples, samplesPerStep, CPUs, batchSize, k):

    toWrite = []
    i = 0
    while i < numSamples:
        samples = recognitionModel.sampleManyHelmholtz(requests, samplesPerStep, CPUs)
        toWrite.extend(samples)
        i = i + len(samples)
        while len(toWrite) > batchSize:
            print("Memory usage: {}".format(getMemoryUsageFraction()))
            batchNum = i // batchSize
            dill.dump(toWrite[:batchSize], open("data/prop_sig/{}_samples_{}/samples_{}-{}.pkl".format(dslName, k, batchSize * (batchNum - 1), batchSize * batchNum), "wb"))
            del toWrite[:batchSize]
            print("Memory usage: {}".format(getMemoryUsageFraction()))

def loadEnumeratedTasks(dslName, k=1, numExamples=11):
    with open("data/prop_sig/{}_enumerated_{}/enumerated_0_10000.pkl".format(dslName, k), "rb") as f:
        frontiers = dill.load(f)

        filteredFrontiers = []
        numTooLong, numWrongType = 0, 0
        for j,f in enumerate(frontiers):
            # assert every frontier is of the desired type
            assert f.task.request == arrow(tlist(tint), tlist(tint))
            # exclude examples where the output is too large

            wrongType = any([(not isinstance(o, list)) or (not isinstance(i[0], list)) for i,o in f.task.examples])
            if wrongType:
                numWrongType += 1
                continue

            examples = [(i,o) for i,o in f.task.examples if len(o) < 50]
            # if sampled task has more than 11 examples keep only the first 11
            if len(examples) < numExamples:
                numTooLong += 1
                continue
            else:
                f.task.examples = examples[:numExamples]
                filteredFrontiers.append(f)

    print("Removed {} tasks cause they had too long outputs".format(numTooLong))
    print("Removed {} tasks cause they were the long type".format(numWrongType))
    return filteredFrontiers

#     bounds = [0.0]
#     while bounds[-1] < upperBound:
#         bounds.append(bounds[-1] + mdlIncrement)

#     allFrontiers = []
#     for lowerBound in bounds:
#         try:
#             path = "data/prop_sig/{}_enumerated_{}/enumerated_{}_{}.pkl".format(dslName, k, str(lowerBound).replace('.', 'p'), str(lowerBound + mdlIncrement).replace('.', 'p'))
#             with open(path, "rb") as f:
#                 frontiers = dill.load(f)
#                 # if sampled task has more than 11 examples keep only the first 11
#                 for j,f in enumerate(frontiers):
#                     numExamples = min(len(f.task.examples), 11)
#                     f.task.examples = f.task.examples[:numExamples]
#                 allFrontiers.extend(frontiers)
#         except Exception as e:
#             print(e)
#             # print("FileNotFoundError for path: {}".format(path))
#             pass

#         if len(allFrontiers) >= n:
#             allFrontiers = allFrontiers[:n]
#             print("Loaded {} tasks with logPrior of at most {}".format(n, lowerBound + mdlIncrement))
#             break

#     if len(allFrontiers) < n:
#         print("loading {} tasks instead of {}".format(len(allFrontiers), n))
#     return allFrontiers

def loadSampledTasks(k=1, batchSize=100, n=10000, dslName="jrule", isSample=True):

    tasksType = "samples" if isSample else "enumerated"
    allFrontiers = []
    for i in range(0,n,batchSize):
            with open("data/prop_sig/{}_{}_{}/{}_{}-{}.pkl".format(dslName, tasksType, k, tasksType, i, i + batchSize), "rb") as f:
                frontiers = dill.load(f)
                # if sampled task has more than 11 examples keep only the first 11
                for j,f in enumerate(frontiers):
                    numExamples = min(len(f.task.examples), 11)
                    f.task.examples = f.task.examples[:numExamples]
                allFrontiers.extend(frontiers)

    # remove the 1981st frontier becuase it is too large
    if k == 1 and dslName == "josh_rich":
        allFrontiers = allFrontiers[:1981] + allFrontiers[1982:]
    return allFrontiers

def getRecognizerTaskGrammars(trainedRecognizer, tasks):
    grammars = {task: trainedRecognizer.grammarOfTask(task)
                for task in tasks}
    #untorch seperately to make sure you filter out None grammars
    grammars = {task: grammar.untorch() for task, grammar in grammars.items() if grammar is not None}
    return grammars

def getPropSimGrammars(baseGrammar, tasks, sampledFrontiers, propertyFeatureExtractor, featureExtractorArgs, onlyUseTrueProperties, nSim, pseudoCounts, weightedSim, compressSimilar, verbose):

    tasksSolved = set()
    grammars = {}
    task2Frontiers = {}

    for task in tasks:
        frontiers,w,solved = getTaskSimilarFrontier(sampledFrontiers, propertyFeatureExtractor, task, baseGrammar, featureExtractorArgs, nSim=nSim, onlyUseTrueProperties=onlyUseTrueProperties, verbose=verbose)
        task2Frontiers[task] = frontiers

        print("---------------------------------------------------------------------------------")
        print(task.describe())
        print("---------------------------------------------------------------------------------")
        
        # equally weight all similar tasks
        for f in frontiers:
            f.entries[0].logPosterior = 0
            f.entries[0].logPrior = 0

        if compressSimilar:
            if len([f for f in frontiers if not f.empty]) == 0:
                eprint("No compression frontiers; not inducing a grammar this iteration.")
            else:

                taskGrammar, compressionFrontiers = induceGrammar(baseGrammar, frontiers,
                                                              topK=1,
                                                              pseudoCounts=pseudoCounts, a=0,
                                                              aic=1.0, structurePenalty=1.5,
                                                              topk_use_only_likelihood=False,
                                                              backend="ocaml", CPUs=1, iteration=0)
                print("Finished inducing grammar")
        else:
            weights = w if weightedSim else None
            taskGrammar = baseGrammar.insideOutside(frontiers, pseudoCounts, iterations=1, frontierWeights=weights)

        grammars[task] = taskGrammar
        if solved:
            tasksSolved.add(task)

    return grammars, tasksSolved, task2Frontiers

def enumerateFromGrammars(grammars, tasks, modelName, enumerationTimeout, solver, CPUs, maximumFrontier, leaveHoldout=False, save=False):
    bottomUpFrontiers, allRecognitionTimes, _, _ = multicoreEnumeration(grammars, tasks, _=None,
                             enumerationTimeout=enumerationTimeout,
                             solver=solver,
                             CPUs=CPUs,
                             maximumFrontier=maximumFrontier,
                             verbose=True,
                             evaluationTimeout=1.0,
                             testing=False,
                             likelihoodModel=None,
                             leaveHoldout=leaveHoldout)
    if save:
        savePath = "enumerationResults/{}_{}_t={}.pkl".format(modelName, datetime.datetime.now(), enumerationTimeout)
        with open(savePath, "wb") as handle:
            dill.dump((bottomUpFrontiers, allRecognitionTimes), handle)
        print("Saved enumeration results at: {}".format(savePath))
    return bottomUpFrontiers, allRecognitionTimes


def comparePropSimFittedToRnnEncoded(train, frontiers, grammar, sampledFrontiers, propertyFeatureExtractor, featureExtractorArgs, nSimList, scoreCutoff, pseudoCounts, compressSimilar):
    """
    Given a frontier of tasks prints out the logposterior of tasks in train using:
        - the RNN-encoded neural recogntion model
        - the unigram grammar fitted on nSim most similar tasks

    """

    uniformGrammarPriors, logVariableGrammarPriors, fittedLogPosteriors, rnnLogPosteriors = 0.0, 0.0, 0.0, 0.0
    fittedLogPosteriorsDict, fittedWeightedLogPosteriorsDict, fittedLogPosteriorsConsolidationDict = {}, {}, {}
    numSolvedPrograms = 0.0
    taskToFrontier = {f.task:f for f in frontiers if len(f.entries) > 0}

    consolidationGrammars, tasksSolved, task2SimFrontiers = getPropSimGrammars(grammar, train, sampledFrontiers, propertyFeatureExtractor, featureExtractorArgs, onlyUseTrueProperties=True, nSim=max(nSimList), pseudoCounts=pseudoCounts, weightedSim=False, compressSimilar=compressSimilar, verbose=False)

    for task in train:

        # can only do analysis if we have program for task from rnn_encoder enumeration
        if task in taskToFrontier:
            bestFrontier = taskToFrontier[task].topK(1)
            if len(bestFrontier) > 0:
                program = bestFrontier.entries[0].program
                logPosterior = bestFrontier.entries[0].logPosterior
                numSolvedPrograms += 1
        else:
            program = None
            continue


        print("\n-------------------------------------------------------------------------------")
        print(task.describe())
        print("---------------------------------------------------------------------------------")
        uniformGrammarPrior = grammar.logLikelihood(task.request, program)
        print("Uniform Grammar Prior: {}".format(uniformGrammarPrior))
        logVariableGrammar = Grammar(2.0, [(0.0, p.infer(), p) for p in grammar.primitives], continuationType=None)
        logVariableGrammarPrior = logVariableGrammar.logLikelihood(task.request, program)
        print("Log Variable Program Prior: {}".format(logVariableGrammarPrior))
        print("---------------------------------------------------------------------------------")


        # if scoreCutoff is not None:
        #     numAboveCutoff = len([w for w in frontierWeights if w >= scoreCutoff])
        #     print("{} simTasks with score >= {}".format(numAboveCutoff, scoreCutoff))

        #     for nSim in nSimList:
        #         taskGrammar = grammar.insideOutside(taskFrontiers[:numAboveCutoff if numAboveCutoff > 0 else nSim], pseudoCounts, iterations=1, frontierWeights=None)
        #         fittedLogPosterior = taskGrammar.logLikelihood(task.request, program)
        #         print("Task Fitted Grammar LP ({} tasks with score > {} / {}): {}".format(numAboveCutoff, scoreCutoff, nSim, fittedLogPosterior))
        #         fittedLogPosteriors100pDict[nSim] = fittedLogPosteriors100pDict.get(nSim, []) + [fittedLogPosterior]

        # for nSim in nSimList:
        #     taskGrammar = grammar.insideOutside(task2SimFrontiers[task][:nSim], pseudoCounts, iterations=1, frontierWeights=None)
        #     fittedLogPosterior = taskGrammar.logLikelihood(task.request, program)
        #     print("Task Fitted Grammar LP ({} weighted frontiers): {}".format(nSim, fittedLogPosterior))
        #     fittedWeightedLogPosteriorsDict[nSim] = fittedWeightedLogPosteriorsDict.get(nSim, []) + [fittedLogPosterior]    

        for nSim in nSimList:
            taskGrammar = grammar.insideOutside(task2SimFrontiers[task][:nSim], pseudoCounts, iterations=1, frontierWeights=None)
            fittedLogPosterior = taskGrammar.logLikelihood(task.request, program)
            print("Task Fitted Grammar LP ({} frontiers): {}".format(nSim, fittedLogPosterior))
            fittedLogPosteriorsDict[nSim] = fittedLogPosteriorsDict.get(nSim, []) + [fittedLogPosterior]

        for nSim in nSimList:
            fittedLogPosterior = consolidationGrammars[task].logLikelihood(task.request, program)
            print("SimTask Consolidation Grammar LP ({} frontiers): {}".format(nSim, fittedLogPosterior))
            fittedLogPosteriorsConsolidationDict[nSim] = fittedLogPosteriorsConsolidationDict.get(nSim, []) + [fittedLogPosterior]

        rnnLogPosterior = bestFrontier.entries[0].logPosterior
        print("RNN Recognition model LP: {}\n".format(rnnLogPosterior))

        uniformGrammarPriors += uniformGrammarPrior
        logVariableGrammarPriors += logVariableGrammarPrior
        rnnLogPosteriors += rnnLogPosterior


    print("Uniform Grammar Prior: {}".format(uniformGrammarPriors / numSolvedPrograms))
    print("Log Variable Grammar Prior: {}".format(logVariableGrammarPriors / numSolvedPrograms))
    print("Mean RNN-Encoded Log Posterior: {}".format(rnnLogPosteriors / numSolvedPrograms))
    # for nSim, logPosteriors in fittedWeightedLogPosteriorsDict.items():
    #     print("Mean Fitted Log Posterior (weighted {} frontiers): {}".format(nSim, sum(logPosteriors) / numSolvedPrograms))
    for nSim, logPosteriors in fittedLogPosteriorsDict.items():
        print("Mean Fitted Log Posterior ({} frontiers): {}".format(nSim, sum(logPosteriors) / numSolvedPrograms))
    for nSim, logPosteriors in fittedLogPosteriorsConsolidationDict.items():
        print("Mean SimTask Consolidation Log Posterior ({} frontiers): {}".format(nSim, sum(logPosteriors) / numSolvedPrograms))
    # for nSim, logPosteriors in fittedLogPosteriors100pDict.items():
    #     print("Mean Fitted Log Posterior (100p, {} deault frontiers): {}".format(nSim, sum(logPosteriors) / numSolvedPrograms))

