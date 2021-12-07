import copy
import datetime
import dill
import json
import numpy as np
import os
import pandas as pd
from threading import Thread,Lock

from dreamcoder.compression import induceGrammar
from dreamcoder.dreaming import helmholtzEnumeration
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.likelihoodModel import UniqueTaskSignatureScore, TaskDiscriminationScore, TaskSurprisalScore, GeneralUniqueTaskSignatureScore
from dreamcoder.program import Program
from dreamcoder.task import Task
from dreamcoder.type import Context, arrow, tlist, tint
from dreamcoder.utilities import *
from dreamcoder.domains.list.property import Property

DATA_DIR = "data/prop_sig/"
MAX_OUTPUT_LENGTH = 1000

def createFrontiersWithInputsFromTask(frontiers, task):
    
    newFrontiers = []
    frontiers = [f for f in frontiers if len(f.entries) > 0]

    for frontier in frontiers:
        func = frontier.entries[0].program.evaluate([])
        try:
            newExamples = []
            for i,_ in task.examples:
                o = task.predict(func, i)
                if len(o) > MAX_OUTPUT_LENGTH:
                    # print("Output too long")
                    # print("Program: {}".format(frontier.entries[0].program))
                    # print("input: {}".format(i))
                    raise ValueError()
                newExamples.append((i,o))
        except Exception as e:
            # print(e)
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


def _makeTaskFromProgram(program, request, featureExtractor, differentOutputs=True, filterIdentityTask=True):
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

def _enumerateFromOcamlGrammar(tasks, grammar, enumerationTimeout, special):
    requests = list({t.request for t in tasks})
    request = requests[0]
    assert len(requests) == 1
    inputs = list({tuplify(xs)
                       for t in tasks if t.request == request
                       for xs, y in t.examples})
    print("Enumerating helmholtz tasks for {} seconds".format(enumerationTimeout))
    response = helmholtzEnumeration(grammar, request, inputs, enumerationTimeout, _=None, special="unique", evaluationTimeout=0.004, maximumSize=99999999)
    return response

def enumerateHelmholtzOcaml(tasks, grammar, enumerationTimeout, CPUs, featureExtractor, save=False, libraryName=None, datasetName=None):
    response = _enumerateFromOcamlGrammar(tasks, grammar, enumerationTimeout, special="unique")
    print("Response length: {}".format(len(response)))
    frontiers = []
    print("First 200 characters of response: {}".format(response[:200]))
    response = json.loads(response.decode("utf-8"))
       
    def parseAndMakeTaskFromProgram(entry, request, featureExtractor):
        program = Program.parse(entry["programs"][0])
        task = _makeTaskFromProgram(program, request, featureExtractor, differentOutputs=True, filterIdentityTask=True)
        if task is None:
            return None
        frontier = Frontier([FrontierEntry(program=Program.parse(p), logPrior=entry["ll"], logLikelihood=0.0) for p in entry["programs"]], task=task)
        return frontier

    requests = list(set([t.request for t in tasks]))
    assert len(requests) == 1
    request = requests[0]
    frontiers = parallelMap(CPUs, lambda entry: parseAndMakeTaskFromProgram(entry, request, featureExtractor), response, memorySensitive=True)
    frontiers = [f for f in frontiers if f is not None] 
    print("{} Frontiers after filtering".format(len(frontiers)))
    
    if save:
        savePath = "{}/helmholtz_frontiers/{}_enumerated/{}_with_{}-inputs.pkl".format(DATA_DIR, libraryName, len(frontiers), datasetName)
        dill.dump(frontiers, open(savePath, "wb"))
        print("Saving frontiers at: {}".format(savePath))

    return frontiers

def enumerateAndSave(grammar, request, featureExtractor, dslName, numTasks, k, batchSize, CPUs=1):

    def enumerateWithinBounds(lowerBound, upperBound, totalNumTasks):
        
        taskCount = sum(totalNumTasks.values())
        print("Have {} valid tasks".format(taskCount))
        if taskCount >= numTasks:
            return

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
                print("{} tasks within {}-{} bounds".format(totalNumTasksEnumerated, lowerBound, upperBound))

        if totalNumTasksEnumerated > 0:
            writePath = "data/prop_sig/{}_enumerated_{}/enumerated_{}_{}.pkl".format(dslName, k, lowerBound, upperBound)
            dill.dump(enumeratedFrontiersBatch, open(writePath, "wb"))
            print("Writing tasks to: {}".format(writePath))
        totalNumTasks[(lowerBound, upperBound)] = totalNumTasksEnumerated
        return

    bounds = [((lowerBound/2.0), (lowerBound/2.0) + 0.5) for lowerBound in range(0,100)]
    totalNumTasks = {bound: 0 for bound in bounds}

    if CPUs > 1:
        parallelMap(CPUs, lambda bounds: enumerateWithinBounds(bounds[0], bounds[1], totalNumTasks), bounds, memorySensitive=True)

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

def loadEnumeratedTasks(filename, primitives=None, numExamples=11):
    
    path = "data/prop_sig/helmholtz_frontiers/{}".format(filename)
    with open(path, "rb") as f:
        frontiers = dill.load(f)

        filteredFrontiers = []
        numTooLong, numWrongType = 0, 0
        for j,f in enumerate(frontiers):

            try:
                p = Program.parse(str(f.topK(1).entries[0].program), primitives={p.name:p for p in primitives})
            except:
                continue

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

            f.task.examples = examples[:numExamples]
            filteredFrontiers.append(f)

    print("Removed {} tasks cause they had too long outputs".format(numTooLong))
    print("Removed {} tasks cause they were the wrong type".format(numWrongType))
    print("{} total frontiers".format(len(filteredFrontiers)))
    return filteredFrontiers


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

