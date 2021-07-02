
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
import torch.nn as nn
import torch

from dreamcoder.dreamcoder import explorationCompression
from dreamcoder.dreaming import helmholtzEnumeration
from dreamcoder.enumeration import multicoreEnumeration
from dreamcoder.likelihoodModel import UniqueTaskSignatureScore, TaskDiscriminationScore, TaskSurprisalScore
from dreamcoder.utilities import eprint, flatten, testTrainSplit, numberOfCPUs, getThisMemoryUsage, getMemoryUsageFraction, howManyGigabytesOfMemory
from dreamcoder.program import *
from dreamcoder.recognition import DummyFeatureExtractor, RecognitionModel
from dreamcoder.grammar import Grammar
from dreamcoder.task import Task
from dreamcoder.type import Context, arrow, tbool, tlist, tint, t0, UnificationFailure
from dreamcoder.domains.list.compareProperties import compare
from dreamcoder.domains.list.propSim import *
from dreamcoder.domains.list.listPrimitives import basePrimitives, primitives, McCarthyPrimitives, bootstrapTarget_extra, no_length, josh_primitives
from dreamcoder.domains.list.makeListTasks import make_list_bootstrap_tasks, sortBootstrap, EASYLISTTASKS, joshTasks
from dreamcoder.domains.list.property import Property
from dreamcoder.domains.list.propertySignatureExtractor import PropertySignatureExtractor
from dreamcoder.domains.list.resultsProcessing import resume_from_path, viewResults
from dreamcoder.domains.list.handwrittenProperties import handWrittenProperties, getHandwrittenPropertiesFromTemplates, tinput, toutput
from dreamcoder.domains.list.utilsBaselines import *
from dreamcoder.domains.list.utilsPlotting import *
from dreamcoder.domains.list.utilsProperties import *
from dreamcoder.domains.list.utilsPropertySampling import *


DATA_DIR = "data/prop_sig/"
SAMPLED_PROPERTIES_DIR = "sampled_properties/"
GRAMMARS_DIR = "grammars/"

def retrieveJSONTasks(filename, features=False):
    """
    For JSON of the form:
        {"name": str,
         "type": {"input" : bool|int|list-of-bool|list-of-int,
                  "output": bool|int|list-of-bool|list-of-int},
         "examples": [{"i": data, "o": data}]}
    """
    with open(filename, "r") as f:
        loaded = json.load(f)
    TP = {
        "bool": tbool,
        "int": tint,
        "list-of-bool": tlist(tbool),
        "list-of-int": tlist(tint),
    }
    return [Task(
        item["name"],
        arrow(TP[item["type"]["input"]], TP[item["type"]["output"]]),
        [((ex["i"],), ex["o"]) for ex in item["examples"]],
        features=None,
        cache=False,
    ) for item in loaded]

try:
    from dreamcoder.recognition import RecurrentFeatureExtractor
    class LearnedFeatureExtractor(RecurrentFeatureExtractor):
        H = 64

        special = None

        def tokenize(self, examples):
            def sanitize(l): return [z if z in self.lexicon else "?"
                                     for z_ in l
                                     for z in (z_ if isinstance(z_, list) else [z_])]

            tokenized = []
            for xs, y in examples:
                if isinstance(y, list):
                    y = ["LIST_START"] + y + ["LIST_END"]
                else:
                    y = [y]
                y = sanitize(y)
                if len(y) > self.maximumLength:
                    return None

                serializedInputs = []
                for xi, x in enumerate(xs):
                    if isinstance(x, list):
                        x = ["LIST_START"] + x + ["LIST_END"]
                    else:
                        x = [x]
                    x = sanitize(x)
                    if len(x) > self.maximumLength:
                        return None
                    serializedInputs.append(x)

                tokenized.append((tuple(serializedInputs), y))

            return tokenized

        def __init__(self, tasks, testingTasks=[], cuda=False, grammar=None, featureExtractorArgs=None):
            self.lexicon = set(flatten((t.examples for t in tasks + testingTasks), abort=lambda x: isinstance(
                x, str))).union({"LIST_START", "LIST_END", "?"})

            # Calculate the maximum length
            self.maximumLength = float('inf') # Believe it or not this is actually important to have here
            self.maximumLength = max(len(l)
                                     for t in tasks + testingTasks
                                     for xs, y in self.tokenize(t.examples)
                                     for l in [y] + [x for x in xs])

            self.parallelTaskOfProgram = True
            self.recomputeTasks = True

            super(
                LearnedFeatureExtractor,
                self).__init__(
                lexicon=list(
                    self.lexicon),
                tasks=tasks,
                cuda=cuda,
                H=self.H,
                bidirectional=True)
except: pass

class CombinedExtractor(nn.Module):
    special = None

    def __init__(self, 
        tasks=[],
        testingTasks=[], 
        cuda=False, 
        H=64, 
        embedSize=16,
        # What should be the timeout for trying to construct Helmholtz tasks?
        helmholtzTimeout=0.25,
        # What should be the timeout for running a Helmholtz program?
        helmholtzEvaluationTimeout=0.01,
        grammar=None,
        featureExtractorArgs=None):
        super(CombinedExtractor, self).__init__()

        self.propSigExtractor = PropertySignatureExtractor(tasks=tasks, testingTasks=testingTasks, H=H, embedSize=embedSize, helmholtzTimeout=helmholtzTimeout, helmholtzEvaluationTimeout=helmholtzEvaluationTimeout,
            cuda=cuda, grammar=grammar, featureExtractorArgs=featureExtractorArgs)
        self.learnedFeatureExtractor = LearnedFeatureExtractor(tasks=tasks, testingTasks=testingTasks, cuda=cuda, grammar=grammar, featureExtractorArgs=featureExtractorArgs)

        # self.propSigExtractor = PropertySignatureExtractor
        # self.learnedFeatureExtractor = LearnedFeatureExtractor

        self.outputDimensionality = H
        self.recomputeTasks = True
        self.parallelTaskOfProgram = True

        self.linear = nn.Linear(2*H, H)

    def forward(self, v, v2=None):
        pass

    def featuresOfTask(self, t):

        learnedFeatureExtractorVector = self.learnedFeatureExtractor.featuresOfTask(t)
        propSigExtractorVector = self.propSigExtractor.featuresOfTask(t)

        if learnedFeatureExtractorVector is not None and propSigExtractorVector is not None:
            return self.linear(torch.cat((learnedFeatureExtractorVector, propSigExtractorVector)))
        else:
            return None

    def featuresOfTasks(self, ts, t2=None):  # Take a task and returns [features]
        """Takes the goal first; optionally also takes the current state second"""
        return [self.featuresOfTask(t) for t in ts]

    def taskOfProgram(self, p, tp):
        return self.learnedFeatureExtractor.taskOfProgram(p=p, tp=tp)


def train_necessary(t):
    if t.name in {"head", "is-primes", "len", "pop", "repeat-many", "tail", "keep primes", "keep squares"}:
        return True
    if any(t.name.startswith(x) for x in {
        "add-k", "append-k", "bool-identify-geq-k", "count-k", "drop-k",
        "empty", "evens", "has-k", "index-k", "is-mod-k", "kth-largest",
        "kth-smallest", "modulo-k", "mult-k", "remove-index-k",
        "remove-mod-k", "repeat-k", "replace-all-with-index-k", "rotate-k",
        "slice-k-n", "take-k",
    }):
        return "some"
    return False


def list_options(parser):
    parser.add_argument(
        "--noMap", action="store_true", default=False,
        help="Disable built-in map primitive")
    parser.add_argument(
        "--noUnfold", action="store_true", default=False,
        help="Disable built-in unfold primitive")
    parser.add_argument(
        "--noLength", action="store_true", default=False,
        help="Disable built-in length primitive")

    # parser.add_argument("--iterations", type=int, default=10)
    # parser.add_argument("--useDSL", action="store_true", default=False)
    parser.add_argument("--split", action="store_true", default=False)
    parser.add_argument("--primitives",  default="property_prims", choices=[
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
            "Lucas-old"])
    parser.add_argument("--extractor", default="prop_sig", choices=[
        "prop_sig",
        "learned",
        "combined",
        "dummy"
        ])
    parser.add_argument("--hidden", type=int, default=64)


    # Arguments relating to propSim
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
    parser.add_argument("--weightedSim", action="store_true", default=False)
    parser.add_argument("--taskSpecificInputs", action="store_true", default=False)
    parser.add_argument("--earlyStopping", action="store_true", default=False)
    parser.add_argument("--singleTask", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--propCPUs", type=int, default=numberOfCPUs())
    parser.add_argument("--propSolver",default="ocaml",type=str)
    parser.add_argument("--propSamplingTimeout",default=600,type=float)
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
    parser.add_argument("--propNoEmbeddings", action="store_true", default=False)


def main(args):
    """
    Takes the return value of the `commandlineArguments()` function as input and
    trains/tests the model on manipulating sequences of numbers.
    """
    
    propNumIters = args.pop("propNumIters")
    hmfSeed = args.pop("hmfSeed")
    numHelmFrontiers = args.pop("numHelmFrontiers")
    maxFractionSame = args.pop("maxFractionSame")
    helmholtzFrontiersFilename = args.pop("helmholtzFrontiersFilename")
    propFilename = args.pop("propFilename")
    propSamplingGrammarWeights = args.pop("propSamplingGrammarWeights")
    filterSimilarProperties = args.pop("filterSimilarProperties")
    computePriorFromTasks = args.pop("computePriorFromTasks")
    nSim = args.pop("nSim")
    propPseudocounts = args.pop("propPseudocounts")
    onlyUseTrueProperties = args.pop("onlyUseTrueProperties")
    weightedSim = args.pop("weightedSim")
    save = args.pop("save")
    verbose = args.pop("verbose")
    libraryName = args.pop("primitives")
    dataset = args.pop("dataset")
    singleTask = args.pop("singleTask")
    debug = args.pop("debug")
    taskSpecificInputs = args.pop("taskSpecificInputs")
    hidden = args.pop("hidden")
    propCPUs = args.pop("propCPUs")
    propSolver = args.pop("propSolver")
    propSamplingTimeout = args.pop("propSamplingTimeout")
    propUseConjunction = args.pop("propUseConjunction")
    propAddZeroToNinePrims = args.pop("propAddZeroToNinePrims")
    propScoringMethod = args.pop("propScoringMethod")
    propDreamTasks = args.pop("propDreamTasks")
    propToUse = args.pop("propToUse")
    propSamplingPrimitives = args.pop("propSamplingPrimitives")
    propNoEmbeddings = args.pop("propNoEmbeddings")


    tasks = {
        "Lucas-old": lambda: retrieveJSONTasks("data/list_tasks.json") + sortBootstrap(),
        "bootstrap": make_list_bootstrap_tasks,
        "sorting": sortBootstrap,
        "Lucas-depth1": lambda: retrieveJSONTasks("data/list_tasks2.json")[:105],
        "Lucas-depth2": lambda: retrieveJSONTasks("data/list_tasks2.json")[:4928],
        "Lucas-depth3": lambda: retrieveJSONTasks("data/list_tasks2.json"),
        "josh_1": lambda: joshTasks("1"),
        "josh_2": lambda: joshTasks("2"),
        "josh_3": lambda: joshTasks("3"),
        "josh_3.1": lambda: joshTasks("3.1"),
        "josh_final": lambda: joshTasks("final"),
    }[dataset]()

    primLibraries = {
             "josh_1": josh_primitives("1"),
             "josh_2": josh_primitives("2"),
             "josh_3": josh_primitives("3")[0],
             "josh_3.1": josh_primitives("3.1")[0],
             "josh_final": josh_primitives("final"),
             "josh_rich": josh_primitives("rich_0_9"),
             "property_prims": handWrittenProperties(),
             "dc_list_domain": bootstrapTarget_extra()
    }

    prims = primLibraries[libraryName]

    if "josh" in dataset:
        tasks = [t for t in tasks if int(t.name[:3]) < 81 and "_1" in t.name]
    tasks = [t for t in tasks if (t.request == arrow(tlist(tint), tlist(tint)) and isinstance(t.examples[0][1],list) and isinstance(t.examples[0][0][0],list))]

    baseGrammar = Grammar.uniform([p for p in prims])
    extractor_name = args.pop("extractor")
    extractor = {
        "dummy": DummyFeatureExtractor,
        "learned": LearnedFeatureExtractor,
        "prop_sig": PropertySignatureExtractor,
        "combined": CombinedExtractor
        }[extractor_name]

    if propSamplingPrimitives != "same":
        propertyPrimitives = primLibraries[propSamplingPrimitives]
    else:
        propertyPrimitives = baseGrammar.primitives

    if extractor_name == "learned":
        featureExtractorArgs = {"hidden":hidden}

    elif extractor_name == "prop_sig" or extractor_name == "combined":
        featureExtractorArgs = {
            "propCPUs": propCPUs,
            "propSolver": propSolver,
            "propSamplingTimeout": propSamplingTimeout,
            "propUseConjunction": propUseConjunction,
            "propAddZeroToNinePrims": propAddZeroToNinePrims,
            "propScoringMethod": propScoringMethod,
            "propDreamTasks": propDreamTasks,
            "propToUse": propToUse,
            "propertyPrimitives": propertyPrimitives,
            "primLibraries": primLibraries,
            "propNoEmbeddings": propNoEmbeddings
        }
        if propToUse == "handwritten":
            properties = getHandwrittenPropertiesFromTemplates(tasks)
            featureExtractor = extractor(tasksToSolve=tasks, allTasks=tasks, grammar=baseGrammar, cuda=False, featureExtractorArgs=featureExtractorArgs, properties=properties)
            print("Loaded {} properties from: {}".format(len(properties), "handwritten"))
        
        elif propToUse == "preloaded":
            assert propFilename is not None
            properties = dill.load(open(DATA_DIR + SAMPLED_PROPERTIES_DIR + propFilename, "rb"))
            if isinstance(properties, dict):
                assert len(properties) == 1
                properties = list(properties.values())[0]
                # filter properties that are only on inputs
                properties = [p for p in properties if "$0" in p.name]
            featureExtractor = extractor(tasksToSolve=tasks, allTasks=tasks, grammar=baseGrammar, cuda=False, featureExtractorArgs=featureExtractorArgs, properties=properties)
            print("Loaded {} properties from: {}".format(len(properties), propFilename))
        
        elif propToUse == "sample":
            # only used if property sampling grammar weights are "fitted"
            fileName = "enumerationResults/neuralRecognizer_2021-05-18 15:27:58.504808_t=600.pkl"
            frontiers, times = dill.load(open(fileName, "rb"))
            allProperties = {}
            tasksToSolve = tasks[0:1]
            returnTypes = [tbool]

            for returnType in returnTypes:
                propertyRequest = arrow(tlist(tint), tlist(tint), returnType)

                grammar = getPropertySamplingGrammar(baseGrammar, propSamplingGrammarWeights, frontiers, pseudoCounts=1, seed=args["seed"])
                try:
                    featureExtractor = extractor(tasksToSolve=tasksToSolve, allTasks=tasks, grammar=grammar, cuda=False, featureExtractorArgs=featureExtractorArgs, propertyRequest=propertyRequest)
                # assertion triggered if 0 properties enumerated
                except AssertionError:
                    print("0 properties found")

                for task in tasksToSolve:
                    allProperties[task] = allProperties.get(task, []) + featureExtractor.properties[task]

            for task in tasksToSolve:

                print("Found {} properties for task {}".format(len(allProperties[task]), task))
                for p in sorted(allProperties[task], key=lambda p: p.score, reverse=True):
                    print("program: {} \nreturnType: {} \nprior: {:.2f} \nscore: {:.2f}".format(p, p.request.returns(), p.logPrior, p.score))
                    print("-------------------------------------------------------------")

            if save:
                filename = "sampled_properties_weights={}_sampling_timeout={}s_return_types={}_seed={}.pkl".format(
                    propSamplingGrammarWeights, int(featureExtractorArgs["propSamplingTimeout"]), returnTypes, args["seed"])
                savePath = DATA_DIR + SAMPLED_PROPERTIES_DIR + filename
                dill.dump(allProperties, open(savePath, "wb"))
                print("Saving sampled properties at: {}".format(savePath))


    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "experimentOutputs/jrule/%s"%timestamp
    # os.system("mkdir -p %s"%outputDirectory)
    
    args.update({
        "featureExtractor": extractor,
        "outputPrefix": "%s/jrule"%outputDirectory,
        "evaluationTimeout": 0.0005,
    })

    if singleTask:
        tasks = [tasks[0]]

    random.seed(args["seed"])
    valuesToInt = {"allFalse":0, "allTrue":1, "mixed":2}

    ##################################
    # Load sampled tasks
    ##################################

    for propSimIteration in range(propNumIters):
        print("\nLoading helmholtz tasks for iteration {}".format(propSimIteration))

        if propSimIteration == 0:
            if helmholtzFrontiersFilename is not None:
                if debug:
                    sampledFrontiers = loadEnumeratedTasks(dslName=libraryName, filename=helmholtzFrontiersFilename, hmfSeed=hmfSeed)
                    randomFrontierIndices = random.sample(range(len(sampledFrontiers)),k=1000)
                    sampledFrontiers = [f for i,f in enumerate(sampledFrontiers) if i in randomFrontierIndices]
                    randomTaskIndices = random.sample(range(len(tasks)),k=1)
                    tasks = [task for i,task in enumerate(tasks) if i in randomTaskIndices]
                else:
                    sampledFrontiers = loadEnumeratedTasks(dslName=libraryName, filename=helmholtzFrontiersFilename, hmfSeed=hmfSeed)
                sampledFrontiers = {t: sampledFrontiers for t in tasks}
            
            task2FittedGrammar = {t:baseGrammar for t in tasks}

        else:
            sampledFrontiers = {}
            for t in tasks:
                sampledFrontiers[t] = enumerateHelmholtzOcaml(tasks, task2FittedGrammar[t], args["enumerationTimeout"], args["CPUs"], featureExtractor, save=save, libraryName=libraryName, dataset=dataset)

        # use subset (numHelmFrontiers) of helmholtz tasks
        for t in tasks:
            if numHelmFrontiers is not None and numHelmFrontiers < len(sampledFrontiers[t]):
                sampledFrontiers[t] = sorted(sampledFrontiers[t], key=lambda f: f.topK(1).entries[0].logPosterior, reverse=True)
                sampledFrontiers[t] = sampledFrontiers[t][:min(len(sampledFrontiers[t]), numHelmFrontiers)]

            print("Finished loading {} helmholtz tasks for task {}\n".format(len(sampledFrontiers[t]), t))

        ##################################
        # Get Grammars
        ##################################

        # helmholtzGrammar = baseGrammar.insideOutside(sampledFrontiers, 1, iterations=1, frontierWeights=None, weightByPrior=False)
        # uniformGrammar = baseGrammar
        # directory = DATA_DIR + "grammars/{}_primitives/enumerated_{}:{}".format(libraryName, hmfSeed, helmholtzFrontiersFilename.split(".")[0])
        # directory += ":{}/".format(numHelmFrontiers) if numHelmFrontiers is not None else "/"
        # neuralGrammars = getGrammarsFromNeuralRecognizer(LearnedFeatureExtractor, tasks, baseGrammar, {"hidden": hidden}, sampledFrontiers, save, directory, args)
        # try:
        #      propSimFilename = "propSim_propToUse={}_nSim={}_weightedSim={}_taskSpecificInputs={}_seed={}.pkl".format(propToUse, nSim, weightedSim, taskSpecificInputs, args["seed"])
        #      path = directory + propSimFilename
        #      propSimGrammars = dill.load(open(path, "rb"))
        # except FileNotFoundError:
             # print("Couldn't find pickled fitted grammars, regenerating")
        task2FittedGrammar, tasksSolved, _ = getPropSimGrammars(
            baseGrammar,
            tasks, 
            sampledFrontiers, 
            featureExtractor, 
            featureExtractorArgs, 
            onlyUseTrueProperties, 
            nSim, 
            propPseudocounts, 
            weightedSim, 
            compressSimilar=False, 
            weightByPrior=False, 
            recomputeTasksWithTaskSpecificInputs=taskSpecificInputs,
            computePriorFromTasks=computePriorFromTasks, 
            filterSimilarProperties=filterSimilarProperties, 
            maxFractionSame=maxFractionSame, 
            valuesToInt=valuesToInt,
            verbose=verbose)

        print("\nSolved {} tasks at iteration {}".format(len(tasksSolved), propSimIteration))

    # if save and not debug:
        # dill.dump(helmholtzGrammar, open(directory + "helmholtzFitted.pkl", "wb"))
        # dill.dump(uniformGrammar, open(directory + "uniformWeights.pkl", "wb"))
       # dill.dump(propSimGrammars, open(directory + propSimFilename, "wb"))

    ##################################
    # Enumeration
    ##################################

    # enumerationTimeout, solver, maximumFrontier, CPUs = args.pop("enumerationTimeout"), args.pop("solver"), args.pop("maximumFrontier"), args.pop("CPUs")

    # for g, modelName in zip(allGrammars, modelNames):
    #      print("grammar for first task: {}".format(g if isinstance(g, Grammar) else list(g.values())[0]))
    #      bottomUpFrontiers, allRecognitionTimes = enumerateFromGrammars(g, tasks, modelName, enumerationTimeout, solver, CPUs, maximumFrontier, leaveHoldout=True, save=save)
    #      nonEmptyFrontiers = [f for f in bottomUpFrontiers if not f.empty]
    #      numTasksSolved = len([f.task for f in nonEmptyFrontiers if f.task.check(f.topK(1).entries[0].program, timeout=1.0, leaveHoldout=False)])
    #      print("Enumerating from {} grammars for {} seconds: {} / {} actually true for holdout example".format(modelName, enumerationTimeout, numTasksSolved, len(nonEmptyFrontiers)))

    #####################
    # Plotting
    #####################

    # filenames = [
    #     "propSim_2021-06-28 22:01:10.416733_t=13200.pkl",
    #     "propSim_2021-06-28 19:33:34.730379_t=1800.pkl",
    #     "neural_2021-06-28 23:20:57.808000_t=13200.pkl",
    #     "neural_2021-06-28 21:14:46.702305_t=1800.pkl",
    #     "helmholtzFitted_2021-06-25 15:36:35.402559_t=600.pkl",
    #     "uniform_2021-06-25 15:47:14.810385_t=600.pkl"
    # ]

    # modelNames = [
    #     "propSim2 (handwritten properties)",
    #     "propSim2 (handwritten properties)",
    #     "neural (RS 10,000)",
    #     "neural (RS 10,000)",
    #     "helmholtzFitted",
    #     "unifGrammarPrior"
    # ]
    
    ######################
    # Enumeration Proxy
    ######################
    
    # fileName = "enumerationResults/propSim_2021-06-23 17:02:48.628976_t=600.pkl"
    # frontiers, times = dill.load(open(fileName, "rb"))
    # unsolvedTasks = [f.task for f in frontiers if len(f.entries) == 0]
    # nSimList = [50]

    # task2FittedGrammars = comparePropSimFittedToRnnEncoded(tasks, frontiers, baseGrammar, sampledFrontiers, featureExtractor, featureExtractorArgs, nSimList, propPseudocounts, 
    #     weightedSim=weightedSim, compressSimilar=False, weightByPrior=False, taskSpecificInputs=taskSpecificInputs, onlyUseTrueProperties=onlyUseTrueProperties, computePriorFromTasks=computePriorFromTasks, 
    #     filterSimilarProperties=filterSimilarProperties, maxFractionSame=maxFractionSame, valuesToInt=valuesToInt, verbose=verbose)

    # if save:
    #     for nSim in nSimList:
    #         propertyInfo = propToUse if propToUse == "handwritten" else propFilename
    #         filename = DATA_DIR + GRAMMARS_DIR + "propToUse={}_nSim={}_weightedSim={}_taskSpecificInputs={}_onlyAllTrue={} \
    #         computePriorFromTasks={}_filterSimilarProperties={}_maxFractionSame={}_seed={}.pkl".format(propToUse, nSim, weightedSim, 
    #             taskSpecificInputs, onlyUseTrueProperties, computePriorFromTasks, filterSimilarProperties, maxFractionSame, args["seed"])
    #         print("Saved task2Grammars at: {}".format(filename))
    #         dill.dump(task2FittedGrammars[nSim], open(filename, "wb"))

