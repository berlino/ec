
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
        "list_prims"])
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
    parser.add_argument("--extractor", default="learned", choices=[
        "prop_sig",
        "learned",
        "combined",
        "dummy"
        ])
    parser.add_argument("--hidden", type=int, default=64)


    # Arguments relating to propSim
    parser.add_argument("--maxFractionSame", type=float, default=1.0)
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
    
    maxFractionSame = args.pop("maxFractionSame")
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

    primLibraries = {"base": basePrimitives,
             "McCarthy": McCarthyPrimitives,
             "common": bootstrapTarget_extra,
             "noLength": no_length,
             "rich": primitives,
             "josh_1": josh_primitives("1"),
             "josh_2": josh_primitives("2"),
             "josh_3": josh_primitives("3")[0],
             "josh_3.1": josh_primitives("3.1")[0],
             "josh_final": josh_primitives("final"),
             "josh_rich": josh_primitives("rich_0_9"),
             "property_prims": handWrittenProperties(),
             "list_prims": bootstrapTarget_extra()
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
            propertyFeatureExtractor = extractor(tasksToSolve=tasks, allTasks=tasks, grammar=baseGrammar, cuda=False, featureExtractorArgs=featureExtractorArgs, properties=properties)
            print("Loaded {} properties from: {}".format(len(properties), "handwritten"))
        
        elif propToUse == "preloaded":
            assert propFilename is not None
            properties = dill.load(open(DATA_DIR + SAMPLED_PROPERTIES_DIR + propFilename, "rb"))
            if isinstance(properties, dict):
                assert len(properties) == 1
                properties = list(properties.values())[0]
                # filter properties that are only on inputs
                properties = [p for p in properties if "$0" in p.name]
            propertyFeatureExtractor = extractor(tasksToSolve=tasks, allTasks=tasks, grammar=baseGrammar, cuda=False, featureExtractorArgs=featureExtractorArgs, properties=properties)
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
                    propertyFeatureExtractor = extractor(tasksToSolve=tasksToSolve, allTasks=tasks, grammar=grammar, cuda=False, featureExtractorArgs=featureExtractorArgs, propertyRequest=propertyRequest)
                # assertion triggered if 0 properties enumerated
                except AssertionError:
                    print("0 properties found")

                for task in tasksToSolve:
                    allProperties[task] = allProperties.get(task, []) + propertyFeatureExtractor.properties[task]

            for task in tasksToSolve:

                print(task.describe())

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


    ##################################
    # Load sampled tasks
    ##################################

    # print("Loading sampled tasks")
    # k = 1
    # # sampledFrontiers = loadSampledTasks(k=k, batchSize=100, n=100, dslName=libraryName, isSample=False)
    # # sampledFrontiers = loadEnumeratedTasks(k=1, mdlIncrement=0.5, n=5000, dslName=libraryName, upperBound=20)

    # if debug:
    #     sampledFrontiers = loadEnumeratedTasks(dslName=libraryName)
    #     randomFrontierIndices = random.sample(range(len(sampledFrontiers)),k=1000)
    #     sampledFrontiers = [f for i,f in enumerate(sampledFrontiers) if i in randomFrontierIndices]

    #     randomTaskIndices = random.sample(range(len(tasks)),k=10)
    #     tasks = [task for i,task in enumerate(tasks) if i in randomTaskIndices]
    # else:
    #     sampledFrontiers = loadEnumeratedTasks(dslName=libraryName)
    # print("Finished loading {} sampled tasks".format(len(sampledFrontiers)))

    ##################################
    # Training Recognition Model
    ##################################

    # recognitionModel = RecognitionModel(
    # featureExtractor=extractor(train, grammar=baseGrammar, testingTasks=[], cuda=torch.cuda.is_available(), featureExtractorArgs=featureExtractorArgs),
    # grammar=baseGrammar,
    # cuda=torch.cuda.is_available(),
    # contextual=False,
    # previousRecognitionModel=False,
    # )

    # # count how many tasks can be tokenized
    # excludeIdx = []
    # for i,f in enumerate(sampledFrontiers):
    #     if recognitionModel.featureExtractor.featuresOfTask(f.task) is None:
    #         excludeIdx.append(i)
    # sampledFrontiers = [f for i,f in enumerate(sampledFrontiers) if i not in excludeIdx]
    # print("Can't get featuresOfTask for {} tasks. Now have {} frontiers".format(len(excludeIdx), len(sampledFrontiers)))

    # ep, CPUs = args.pop("earlyStopping"), args.pop("CPUs")
    # recognitionSteps = args.pop("recognitionSteps")
    # recognitionTimeout = args.pop("recognitionTimeout")
    # trainedRecognizer = recognitionModel.trainRecognizer(
    # frontiers=sampledFrontiers, 
    # helmholtzFrontiers=[],
    # helmholtzRatio=args.pop("helmholtzRatio"),
    # CPUs=CPUs,
    # lrModel=False, 
    # earlyStopping=ep, 
    # holdout=ep,
    # steps=recognitionSteps,
    # timeout=recognitionTimeout,
    # defaultRequest=arrow(tlist(tint), tlist(tint)))

    # path = "recognitionModels/{}_enumerated_{}/learned_{}_enumeratedFrontiers_ep={}_RS={}_RT={}.pkl".format(libraryName, k, len(sampledFrontiers), ep, recognitionSteps, recognitionTimeout)
    # with open(path,'wb') as handle:
    #     print("Saved recognizer at: {}".format(path))
    #     dill.dump(trainedRecognizer, handle)

    # loadPath = "enumerationResults/learned_2021-05-10 13:34:42.593000_t=1.pkl"
    # with open(loadPath, "rb") as handle:
    #     frontiers, recognitionTimes = dill.load(handle)

    ##################################
    # Enumeration
    ##################################
    # try:
    #     filename = "propToUse={}_nSim={}_weightedSim={}_taskSpecificInputs={}_seed={}.pkl".format(propToUse, nSim, weightedSim, taskSpecificInputs, args["seed"])
    #     path = DATA_DIR + GRAMMARS_DIR + filename
    #     propSimGrammars = dill.load(open(path, "rb"))
    # except FileNotFoundError:
    #     print("Couldn't find pickled fitted grammars, regenerating")
    #     propSimGrammars, tasksSolved, _ = getPropSimGrammars(baseGrammar, tasks, sampledFrontiers, propertyFeatureExtractor, featureExtractorArgs, onlyUseTrueProperties, [nSim], propPseudocounts, weightedSim, 
    #     compressSimilar=False, weightByPrior=False, recomputeTasksWithTaskSpecificInputs=False, verbose=verbose)

    # enumerationTimeout, solver, maximumFrontier, CPUs = args.pop("enumerationTimeout"), args.pop("solver"), args.pop("maximumFrontier"), args.pop("CPUs")
    # modelName = "propSim"
    # grammars = propSimGrammars[nSim]

    # bottomUpFrontiers, allRecognitionTimes = enumerateFromGrammars(grammars, tasks, modelName, enumerationTimeout, solver, CPUs, maximumFrontier, leaveHoldout=True, save=save)
    # nonEmptyFrontiers = [f for f in bottomUpFrontiers if not f.empty]
    # numTasksProgramDiscovered = len(nonEmptyFrontiers)
    # numTasksSolved = len([f.task for f in nonEmptyFrontiers if f.task.check(f.topK(1).entries[0].program, timeout=1.0, leaveHoldout=False)])
    # print("Enumerating from {} grammars for {} seconds: {} / {} actually true for holdout example".format(modelName, enumerationTimeout, numTasksProgramDiscovered, numTasksSolved, numTasksProgramDiscovered))
#
    #####################
    # Helmhholtz Sampling
    #####################

    # k = 2
    # sampleAndSave(recognitionModel, [arrow(tlist(tint), tlist(tint))], dslName=libraryName, numSamples=10000, samplesPerStep=1000, CPUs=40, batchSize=100, k=k)
    # f = loadSampledTasks(k=2, batchSize=2, n=4, dslName=libraryName)
    # print(f)
    
    # featureExtractor=extractor(train, grammar=baseGrammar, testingTasks=[], cuda=torch.cuda.is_available(), featureExtractorArgs=featureExtractorArgs)
    # enumerateAndSave(baseGrammar, train[0].request, featureExtractor, dslName=libraryName, numTasks=10000, k=1, batchSize=100, CPUs=args.pop("CPUs"))

    #####################
    # Plotting
    #####################

    # filenames = [
    # "enumerationResults/neuralRecognizer_2021-05-23 04:43:46.411556_t=600.pkl",
    # "enumerationResults/propSim_2021-05-23 04:57:26.284483_t=600.pkl",
    # "enumerationResults/propSim_propToUse=sampled_properties_fitted_60s_sampling_timeout.pkl_nSim=50_weightedSim=False_taskSpecificInputs=False_onlyAllTrue=False_seed=1.pkl_2021-06-18 15:34:26.870940_t=600.pkl",
    # "enumerationResults/propSim_propToUse=handwritten_equivalent_sampled_properties_fitted_60s_sampling_timeout.pkl_nSim=50_weightedSim=False_taskSpecificInputs=False_onlyAllTrue=False_seed=1.pkl_2021-06-18 15:09:48.214477_t=600.pkl",
    # "enumerationResults/helmholtz_fitted_2021-06-11 13:17:47.928151_t=600.pkl",
    # "enumerationResults/uniformGrammar_2021-05-11 13:49:32.922951_t=600.pkl"]
    # modelNames = [
    # "neuralRecognizer",
    # "propSim2 (handwritten properties)",
    # "propSim2 (fitted sampled properties)",
    # "propSim2 (fitted sampled only handwritten equivalent properties)",
    # "helmholtzFitted",
    # "unifGrammarPrior"]
    # plotFrontiers(filenames, modelNames)
    
    ######################
    # Enumeration Proxy
    ######################
    
    # fileName = "enumerationResults/propSim_2021-05-23 04:57:26.284483_t=600.pkl"
    # frontiers, times = dill.load(open(fileName, "rb"))
    # unsolvedTasks = [f.task for f in frontiers if len(f.entries) == 0]
    # nSimList = [50]
    # valuesToInt = {"allFalse":0, "allTrue":1, "mixed":2}

    # task2FittedGrammars = comparePropSimFittedToRnnEncoded(tasks, frontiers, baseGrammar, sampledFrontiers, propertyFeatureExtractor, featureExtractorArgs, nSimList, propPseudocounts, 
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

    # propertyFeatureExtractor = extractor([f.task for f in sampledFrontiers], grammar=baseGrammar, cuda=False, featureExtractorArgs=featureExtractorArgs, properties=properties)
    # onlySampleFor100percentSimTasks = True
    # for i,task in enumerate(train):
    #     similarTaskFrontiers, frontierWeights, solved = getTaskSimilarFrontier(sampledFrontiers, propertyFeatureExtractor, task, baseGrammar, featureExtractorArgs, nSim=5, onlyUseTrueProperties=True, verbose=True)
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

    ########################################################################################################
    # Enumerate from holes
    ########################################################################################################
    # program = Program.parse("(lambda (insert 0 1 $0))", primitives={p.name:p for p in baseGrammar.primitives})
    # holes = baseGrammar.enumerateHoles(train[0].request, program)
    # sketch = holes[0][0]

    # solution = Program.parse("(lambda (insert (if (gt? (length $0) 4) 5 8) 1 $0))", primitives={p.name:p for p in baseGrammar.primitives})
    # print(baseGrammar.logLikelihood(train[0].request, program))
    # print(baseGrammar.logLikelihood(train[0].request, solution))
    # print(fittedGrammar.logLikelihood(train[0].request, solution))

    # frontiers, bestSearchTime, taskToNumberOfPrograms, likelihoodModel = multicoreEnumeration(baseGrammar, unsolvedTasks, _=None,
    #                      enumerationTimeout=600,
    #                      solver="python",
    #                      CPUs=1,
    #                      maximumFrontier=5,
    #                      verbose=True,
    #                      evaluationTimeout=1.0,
    #                      testing=False,
    #                      likelihoodModel=None,
    #                      leaveHoldout=False,
    #                      similarTasks=None,
    #                      enumerateFromSketch=sketch)

    ########################################################################################################
    # Enumerate Tasks
    ########################################################################################################

    request = list({t.request for t in tasks})[0]
    inputs = list({tuplify(xs)
                       for t in tasks if t.request == request
                       for xs, y in t.examples})

              
    frontiers = helmholtzEnumeration(baseGrammar, request, inputs, args["enumerationTimeout"], _=None, special="unique", evaluationTimeout=0.1)
    print(frontiers)

        
    # print(frontiers)
    # explorationCompression(baseGrammar, train, testingTasks=test, featureExtractorArgs=featu
