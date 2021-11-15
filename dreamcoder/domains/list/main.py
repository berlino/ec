
from collections import defaultdict
import copy
import datetime
import dill
import json
import math
import numpy as np
import os
import pandas as pd
import random

from dreamcoder.dreamcoder import explorationCompression
from dreamcoder.dreaming import helmholtzEnumeration
from dreamcoder.enumeration import multicoreEnumeration
from dreamcoder.likelihoodModel import UniqueTaskSignatureScore, TaskDiscriminationScore, TaskSurprisalScore
from dreamcoder.program import *
from dreamcoder.grammar import Grammar
from dreamcoder.domains.list.compareProperties import compare
from dreamcoder.domains.list import propSimMain
from dreamcoder.domains.list.property import Property
from dreamcoder.domains.list.resultsProcessing import resume_from_path, viewResults
from dreamcoder.domains.list.runUtils import *
from dreamcoder.domains.list.utilsPlotting import *
from dreamcoder.domains.list.utilsPropertySampling import *

def main(args):
    """
    Takes the return value of the `
    guments()` function as input and
    trains/tests the model on manipulating sequences of numbers.
    """
    
    extractorName = args.pop("extractor")
    propNumIters = args.pop("propNumIters")
    hmfSeed = args.pop("hmfSeed")
    helmholtzFrontiersFilename = args.pop("helmholtzFrontiersFilename")
    propFilename = args.pop("propFilename")
    propSamplingGrammarWeights = args.pop("propSamplingGrammarWeights")
    save = args.pop("save")
    libraryName = args.pop("libraryName")
    dataset = args.pop("dataset")
    singleTask = args.pop("singleTask")
    debug = args.pop("debug")
    hidden = args.pop("hidden")
    propCPUs = args.pop("propCPUs")
    propSolver = args.pop("propSolver")
    propEnumerationTimeout = args.pop("propEnumerationTimeout")
    propUseConjunction = args.pop("propUseConjunction")
    propAddZeroToNinePrims = args.pop("propAddZeroToNinePrims")
    propScoringMethod = args.pop("propScoringMethod")
    propDreamTasks = args.pop("propDreamTasks")
    propToUse = args.pop("propToUse")
    propSamplingPrimitives = args.pop("propSamplingPrimitives")
    propUseEmbeddings = args.pop("propUseEmbeddings")

    numHelmFrontiers = args["numHelmFrontiers"]
    onlyUseTrueProperties = args["onlyUseTrueProperties"]
    nSim = args["nSim"]
    propPseudocounts = args["propPseudocounts"]
    weightedSim = args["weightedSim"]
    weightByPrior= args["weightByPrior"]
    taskSpecificInputs = args["taskSpecificInputs"]
    computePriorFromTasks = args["computePriorFromTasks"]
    filterSimilarProperties = args["filterSimilarProperties"]
    maxFractionSame = args["maxFractionSame"]
    verbose = args["verbose"]
    valuesToInt = {"allFalse":0, "allTrue":1, "mixed":2}

    tasks = get_tasks(dataset) 
    tasks = tasks[0:1] if singleTask else tasks
    extractor = get_extractor(extractorName)
    prims = get_primitives(libraryName)
    baseGrammar = Grammar.uniform([p for p in prims])

    if propSamplingPrimitives != "same":
        propertyPrimitives = primLibraries[propSamplingPrimitives]
    else:
        propertyPrimitives = baseGrammar.primitives

    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "experimentOutputs/jrule/%s/"%timestamp
    os.system("mkdir -p %s"%outputDirectory)
    
    args.update({
        "featureExtractor": extractor,
        "outputPrefix": "jrule",
        "outputDirectory": outputDirectory,
        "evaluationTimeout": 0.0005,
        "valuesToInt": valuesToInt
    })

    random.seed(args["seed"])

    explorationCompression(baseGrammar, tasks, testingTasks=[], featureExtractorArgs=featureExtractorArgs, **args)


