
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

def list_options(parser):

    # parser.add_argument("--iterations", type=int, default=10)
    # parser.add_argument("--useDSL", action="store_true", default=False)
    parser.add_argument("--libraryName",  default="property_prims", choices=[
        "josh_1",
        "josh_2",
        "josh_3",
        "josh_3.1",
        "josh_final",
        "josh_rich",
        "property_prims",
        "dc_list_domain"])
    parser.add_argument("--propSamplingPrimitives", default="same", choices=[
        "same",
        "josh_1",
        "josh_2",
        "josh_3",
        "josh_3.1",
        "josh_final",
        "josh_rich",
        "property_prims",
        "list_prims"])
    parser.add_argument(
        "--dataset",
        type=str,
        default="josh_3",
        choices=[
            "josh_1",
            "josh_2",
            "josh_3",
            "josh_3.1",
            "josh_final",
            "josh_fleet",
            "josh_fleet0to9",
            "Lucas-old"])
    parser.add_argument("--extractor", default="prop_sig", choices=[
        "prop_sig",
        "learned",
        "combined",
        "dummy"
        ])
    parser.add_argument("--hidden", type=int, default=64)

    # Arguments relating to propSim
    parser.add_argument("--propSim", action="store_true", default=False)
    parser.add_argument("--helmEnumerationTimeout", type=int, default=1)
    parser.add_argument("--propNumIters", type=int, default=1)
    parser.add_argument("--hmfSeed", type=int, default=1)
    parser.add_argument("--numHelmFrontiers", type=int, default=None)
    parser.add_argument("--maxFractionSame", type=float, default=1.0)
    parser.add_argument("--helmholtzFrontiersFilename", type=str, default=None)
    parser.add_argument("--propFilename", type=str, default=None)
    parser.add_argument("--filterSimilarProperties", action="store_true", default=False)
    parser.add_argument("--computePriorFromTasks", action="store_true", default=False)
    parser.add_argument("--nSim", type=int, default=50)
    parser.add_argument("--propPseudocounts", type=int, default=1)
    parser.add_argument("--onlyUseTrueProperties", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--weightByPrior", action="store_true", default=False)
    parser.add_argument("--weightedSim", action="store_true", default=False)
    parser.add_argument("--taskSpecificInputs", action="store_true", default=False)
    parser.add_argument("--earlyStopping", action="store_true", default=False)
    parser.add_argument("--singleTask", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--propCPUs", type=int, default=numberOfCPUs())
    parser.add_argument("--propSolver",default="ocaml",type=str)
    parser.add_argument("--propEnumerationTimeout",default=1,type=float)
    parser.add_argument("--propUseConjunction", action="store_true", default=False)
    parser.add_argument("--propAddZeroToNinePrims", action="store_true", default=False)
    parser.add_argument("--propScoringMethod", default="unique_task_signature", choices=[
        "per_task_discrimination",
        "unique_task_signature",
        "general_unique_task_signature",
        "per_similar_task_discrimination",
        "per_task_surprisal"
        ])
    parser.add_argument("--propDreamTasks", action="store_true", default=False)
    parser.add_argument("--propToUse", default="handwritten", choices=[
        "handwritten",
        "preloaded",
        "sample"
        ])
    parser.add_argument("--propSamplingGrammarWeights", default="uniform", choices=[
        "uniform",
        "fitted",
        "random"
        ])
    parser.add_argument("--propUseEmbeddings", action="store_true", default=False)


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


