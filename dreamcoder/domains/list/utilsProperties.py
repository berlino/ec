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
from dreamcoder.likelihoodModel import UniqueTaskSignatureScore, TaskDiscriminationScore, TaskSurprisalScore, GeneralUniqueTaskSignatureScore
from dreamcoder.task import Task
from dreamcoder.type import Context, arrow, tlist, tint
from dreamcoder.utilities import *
from dreamcoder.domains.list.property import Property
from dreamcoder.domains.list.propSim import getPropSimGrammars


def createFrontiersWithInputsFromTask(frontiers, task):
    
    newFrontiers = []
    frontiers = [f for f in frontiers if len(f.entries) > 0]

    for frontier in frontiers:
        func = frontier.entries[0].program.evaluate([])
        try:
            newExamples = [(i,task.predict(func,i)) for i,o in task.examples]
        except:
            continue
        newTask = Task(frontier.task.name, frontier.task.request, newExamples, features=None, cache=False)
        newFrontier = Frontier(frontier.entries, newTask)
        newFrontiers.append(newFrontier)

    print("Dropping {} out of {} due to error when executing on specified inputs".format(len(frontiers) - len(newFrontiers), len(frontiers)))
    return newFrontiers


def _reduceByEvaluating(p, primitives):
    evalReduced, p = p.evalReduce(primitives)
    while evalReduced:
        evalReduced, temp = p.evalReduce(primitives)
        if temp == p:
            break
        else:
            p = temp
    return p

def convertToPropertyTasks(tasks, propertyRequest):
    propertyTasks = []
    for i,t in enumerate(tasks):
        # if i == 1981:
        #     continue
        tCopy = copy.deepcopy(t)
        tCopy.specialTask = ("property", None)
        tCopy.request = propertyRequest
        # tCopy.examples = [io for io in tCopy.examples]
        # tCopy.examples = [(tuplify([io[0][0], (io[1],)]), True) for io in tCopy.examples]
        propertyTasks.append(tCopy)
    return propertyTasks


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
    print("Removed {} tasks cause they were the wrong type".format(numWrongType))
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


def comparePropSimFittedToRnnEncoded(train, frontiers, grammar, sampledFrontiers, propertyFeatureExtractor, featureExtractorArgs, nSimList, pseudoCounts, weightedSim, compressSimilar, weightByPrior, taskSpecificInputs, onlyUseTrueProperties, computePriorFromTasks, verbose=False):
    """
    Given a frontier of tasks prints out the logposterior of tasks in train using:
        - the RNN-encoded neural recogntion model
        - the unigram grammar fitted on nSim most similar tasks
    """

    uniformGrammarPriors, logVariableGrammarPriors, fittedLogPosteriors, baselineLogPosteriors = 0.0, 0.0, 0.0, 0.0
    fittedLogPosteriorsDict, fittedWeightedLogPosteriorsDict, fittedLogPosteriorsConsolidationDict = {}, {}, {}
    taskToFrontier = {f.task:f for f in frontiers if len(f.entries) > 0}

    task2FittedGrammars, tasksSolved, task2SimFrontiers = getPropSimGrammars(grammar, train, sampledFrontiers, propertyFeatureExtractor, featureExtractorArgs, 
        onlyUseTrueProperties=onlyUseTrueProperties, nSimList=nSimList, pseudoCounts=pseudoCounts, weightedSim=weightedSim, compressSimilar=False, weightByPrior=weightByPrior, 
        recomputeTasksWithTaskSpecificInputs=taskSpecificInputs, computePriorFromTasks=computePriorFromTasks, verbose=verbose)

    if compressSimilar:
        raise NotImplementedError

    numTasks = 0
    for task in train:

        if task in taskToFrontier:
            bestFrontier = taskToFrontier[task].topK(1)
            program = bestFrontier.entries[0].program
            logPosterior = bestFrontier.entries[0].logPosterior
            numTasks += 1
        else:
            continue


        vprint("\n-------------------------------------------------------------------------------", verbose)
        vprint(task.describe(), verbose)
        vprint("---------------------------------------------------------------------------------", verbose)
        uniformGrammarPrior = grammar.logLikelihood(task.request, program)
        vprint("Uniform Grammar Prior: {}".format(uniformGrammarPrior), verbose)
        logVariableGrammar = Grammar(2.0, [(0.0, p.infer(), p) for p in grammar.primitives], continuationType=None)
        logVariableGrammarPrior = logVariableGrammar.logLikelihood(task.request, program)
        vprint("Log Variable Program Prior: {}".format(logVariableGrammarPrior), verbose)
        vprint("---------------------------------------------------------------------------------", verbose)


        for nSim in nSimList:
            fittedLogPosterior = task2FittedGrammars[nSim][task].logLikelihood(task.request, program)
            vprint("PropSim Grammar LP ({} frontiers): {}".format(nSim, fittedLogPosterior), verbose)
            fittedLogPosteriorsDict[nSim] = fittedLogPosteriorsDict.get(nSim, []) + [fittedLogPosterior]

        if compressSimilar:
            raise NotImplementedError
            # for nSim in nSimList:
            #     fittedLogPosterior = consolidationGrammars[task].logLikelihood(task.request, program)
            #     print("PropSim Consolidation Grammar LP ({} frontiers): {}".format(nSim, fittedLogPosterior))
            #     fittedLogPosteriorsConsolidationDict[nSim] = fittedLogPosteriorsConsolidationDict.get(nSim, []) + [fittedLogPosterior]

        baselineLogPosterior = bestFrontier.entries[0].logPosterior
        vprint("Baseline LogPosterior: {}\n".format(baselineLogPosterior), verbose)

        uniformGrammarPriors += uniformGrammarPrior
        logVariableGrammarPriors += logVariableGrammarPrior
        baselineLogPosteriors += logPosterior

    if numTasks == 0:
        print("No solved frontiers from which to report metrics")
        return
    else:
        print("Metrics for {} tasks with solutions".format(numTasks))

    print("Mean Uniform Grammar Prior: {}".format(uniformGrammarPriors / numTasks))
    print("Mean Log Variable Grammar Prior: {}".format(logVariableGrammarPriors / numTasks))
    print("Mean Baseline Log Posterior: {}".format(baselineLogPosteriors / numTasks))
    for nSim, logPosteriors in fittedLogPosteriorsDict.items():
        print("Mean Fitted Log Posterior ({} frontiers): {}".format(nSim, sum(logPosteriors) / numTasks))
    if compressSimilar:
        for nSim, logPosteriors in fittedLogPosteriorsConsolidationDict.items():
            print("Mean SimTask Consolidation Log Posterior ({} frontiers): {}".format(nSim, sum(logPosteriors) / numTasks))

    return task2FittedGrammars

