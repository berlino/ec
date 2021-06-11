import random
from collections import defaultdict
import copy
import json
import math
import numpy as np
import os
import pandas as pd
import torch.nn as nn
import torch
import datetime
import dill

from dreamcoder.dreamcoder import explorationCompression
from dreamcoder.enumeration import multicoreEnumeration
from dreamcoder.likelihoodModel import UniqueTaskSignatureScore, TaskDiscriminationScore
from dreamcoder.utilities import eprint, flatten, testTrainSplit, numberOfCPUs, getThisMemoryUsage, getMemoryUsageFraction, howManyGigabytesOfMemory
from dreamcoder.program import *
from dreamcoder.recognition import DummyFeatureExtractor, RecognitionModel
from dreamcoder.grammar import Grammar
from dreamcoder.task import Task
from dreamcoder.type import Context, arrow, tbool, tlist, tint, t0, UnificationFailure
from dreamcoder.domains.list.listPrimitives import basePrimitives, primitives, McCarthyPrimitives, bootstrapTarget_extra, no_length, josh_primitives
from dreamcoder.domains.list.makeListTasks import make_list_bootstrap_tasks, sortBootstrap, EASYLISTTASKS, joshTasks
from dreamcoder.domains.list.property import Property
from dreamcoder.domains.list.propertySignatureExtractor import PropertySignatureExtractor
from dreamcoder.domains.list.resultsProcessing import resume_from_path, viewResults, plotFrontiers
from dreamcoder.domains.list.taskProperties import handWrittenProperties, getHandwrittenPropertiesFromTemplates, tinput, toutput
from dreamcoder.domains.list.utilsProperties import *
from dreamcoder.domains.list.utilsPropertySampling import updateSavedPropertiesWithNewCacheTable


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
        features=(None if not features else list_features(
            [((ex["i"],), ex["o"]) for ex in item["examples"]])),
        cache=False,
    ) for item in loaded]


def list_features(examples):
    if any(isinstance(i, int) for (i,), _ in examples):
        # obtain features for number inputs as list of numbers
        examples = [(([i],), o) for (i,), o in examples]
    elif any(not isinstance(i, list) for (i,), _ in examples):
        # can't handle non-lists
        return []
    elif any(isinstance(x, list) for (xs,), _ in examples for x in xs):
        # nested lists are hard to extract features for, so we'll
        # obtain features as if flattened
        examples = [(([x for xs in ys for x in xs],), o)
                    for (ys,), o in examples]

    # assume all tasks have the same number of examples
    # and all inputs are lists
    features = []
    ot = type(examples[0][1])

    def mean(l): return 0 if not l else sum(l) / len(l)
    imean = [mean(i) for (i,), o in examples]
    ivar = [sum((v - imean[idx])**2
                for v in examples[idx][0][0])
            for idx in range(len(examples))]

    # DISABLED length of each input and output
    # total difference between length of input and output
    # DISABLED normalized count of numbers in input but not in output
    # total normalized count of numbers in input but not in output
    # total difference between means of input and output
    # total difference between variances of input and output
    # output type (-1=bool, 0=int, 1=list)
    # DISABLED outputs if integers, else -1s
    # DISABLED outputs if bools (-1/1), else 0s
    if ot == list:  # lists of ints or bools
        omean = [mean(o) for (i,), o in examples]
        ovar = [sum((v - omean[idx])**2
                    for v in examples[idx][1])
                for idx in range(len(examples))]

        def cntr(
            l, o): return 0 if not l else len(
            set(l).difference(
                set(o))) / len(l)
        cnt_not_in_output = [cntr(i, o) for (i,), o in examples]

        #features += [len(i) for (i,), o in examples]
        #features += [len(o) for (i,), o in examples]
        features.append(sum(len(i) - len(o) for (i,), o in examples))
        #features += cnt_not_int_output
        features.append(sum(cnt_not_in_output))
        features.append(sum(om - im for im, om in zip(imean, omean)))
        features.append(sum(ov - iv for iv, ov in zip(ivar, ovar)))
        features.append(1)
        # features += [-1 for _ in examples]
        # features += [0 for _ in examples]
    elif ot == bool:
        outs = [o for (i,), o in examples]

        #features += [len(i) for (i,), o in examples]
        #features += [-1 for _ in examples]
        features.append(sum(len(i) for (i,), o in examples))
        #features += [0 for _ in examples]
        features.append(0)
        features.append(sum(imean))
        features.append(sum(ivar))
        features.append(-1)
        # features += [-1 for _ in examples]
        # features += [1 if o else -1 for o in outs]
    else:  # int
        def cntr(
            l, o): return 0 if not l else len(
            set(l).difference(
                set(o))) / len(l)
        cnt_not_in_output = [cntr(i, [o]) for (i,), o in examples]
        outs = [o for (i,), o in examples]

        #features += [len(i) for (i,), o in examples]
        #features += [1 for (i,), o in examples]
        features.append(sum(len(i) for (i,), o in examples))
        #features += cnt_not_int_output
        features.append(sum(cnt_not_in_output))
        features.append(sum(o - im for im, o in zip(imean, outs)))
        features.append(sum(ivar))
        features.append(0)
        # features += outs
        # features += [0 for _ in examples]

    return features


def isListFunction(tp):
    try:
        Context().unify(tp, arrow(tlist(tint), t0))
        return True
    except UnificationFailure:
        return False


def isIntFunction(tp):
    try:
        Context().unify(tp, arrow(tint, t0))
        return True
    except UnificationFailure:
        return False

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
        "combined"
        ])
    parser.add_argument("--hidden", type=int, default=64)


    # Arguments relating to properties
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
        "per_similar_task_discrimination"
        ])
    parser.add_argument("--propDreamTasks", action="store_true", default=False)
    parser.add_argument("--propToUse", default="handwritten", choices=[
        "handwritten",
        "preloaded",
        "sample"
        ])
    parser.add_argument("--propNoEmbeddings", action="store_true", default=False)


def main(args):
    """
    Takes the return value of the `commandlineArguments()` function as input and
    trains/tests the model on manipulating sequences of numbers.
    """
    
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
        elif propToUse == "preloaded":
            propertiesFilename = "sampled_properties_fitted_60s_sampling_timeout.pkl"
            properties = dill.load(open(DATA_DIR + SAMPLED_PROPERTIES_DIR + propertiesFilename, "rb"))
        propertyFeatureExtractor = extractor(tasks, grammar=baseGrammar, cuda=False, featureExtractorArgs=featureExtractorArgs, properties=properties)

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

    print("Loading sampled tasks")
    k = 1
    # sampledFrontiers = loadSampledTasks(k=k, batchSize=100, n=100, dslName=libraryName, isSample=False)
    # sampledFrontiers = loadEnumeratedTasks(k=1, mdlIncrement=0.5, n=5000, dslName=libraryName, upperBound=20)

    if debug:
        sampledFrontiers = loadEnumeratedTasks(dslName=libraryName)
        randomFrontierIndices = random.sample(range(len(sampledFrontiers)),k=100)
        sampledFrontiers = [f for i,f in enumerate(sampledFrontiers) if i in randomFrontierIndices]

        randomTaskIndices = random.sample(range(len(tasks)),k=2)
        tasks = [task for i,task in enumerate(tasks) if i in randomTaskIndices]
    else:
        sampledFrontiers = loadEnumeratedTasks(dslName=libraryName)
    print("Finished loading {} sampled tasks".format(len(sampledFrontiers)))

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
    nSim, pseudoCounts, onlyUseTrueProperties = 50, 1, True
    try:
        filename = DATA_DIR + GRAMMARS_DIR + "propToUse={}_nSim={}_weightedSim={}_taskSpecificInputs={}.pkl".format(propToUse, nSim, weightedSim, taskSpecificInputs)
        propSimGrammars = dill.load(open(filename, "rb"))
    except FileNotFoundError:
        print("Couldn't find pickled fitted grammars, regenerating")
        propSimGrammars, tasksSolved, _ = getPropSimGrammars(baseGrammar, tasks, sampledFrontiers, propertyFeatureExtractor, featureExtractorArgs, onlyUseTrueProperties, [nSim], pseudoCounts, weightedSim, 
            compressSimilar=False, weightByPrior=False, recomputeTasksWithTaskSpecificInputs=False, verbose=False)

    enumerationTimeout, solver, maximumFrontier, CPUs = args.pop("enumerationTimeout"), args.pop("solver"), args.pop("maximumFrontier"), args.pop("CPUs")
    modelName = "propSim"
    grammars = propSimGrammars[nSim]

    bottomUpFrontiers, allRecognitionTimes = enumerateFromGrammars(grammars, tasks, modelName, enumerationTimeout, solver, CPUs, maximumFrontier, leaveHoldout=True, save=save)
    nonEmptyFrontiers = [f for f in bottomUpFrontiers if not f.empty]
    numTasksProgramDiscovered = len(nonEmptyFrontiers)
    numTasksSolved = len([f.task for f in nonEmptyFrontiers if f.task.check(f.topK(1).entries[0].program, timeout=1.0, leaveHoldout=False)])
    print("Enumerating from {} grammars for {} seconds: {} / {} actually true for holdout example".format(modelName, enumerationTimeout, numTasksProgramDiscovered, numTasksSolved, numTasksProgramDiscovered))
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
    # # "enumerationResults/neuralRecognizer_2021-05-11 15:23:26.568699_t=600.pkl", "enumerationResults/neuralRecognizer_2021-05-11 18:04:39.262428_t=600.pkl", 
    # # "enumerationResults/neuralRecognizer_2021-05-17 15:54:55.857341_t=600.pkl",
    # "enumerationResults/neuralRecognizer_2021-05-23 04:43:46.411556_t=600.pkl",
    # # "enumerationResults/neuralRecognizer_2021-05-23 04:43:46.411556_t=600.pkl",
    # # "enumerationResults/propSim_2021-05-10 15:28:36.921856_t=600.pkl", "enumerationResults/propSim_2021-05-11 12:49:11.749481_t=600.pkl", 
    # # "enumerationResults/propSim_2021-05-12 22:15:52.858092_t=600.pkl", "enumerationResults/propSim_2021-05-12 22:42:53.788790_t=600.pkl", 
    # "enumerationResults/propSim_2021-05-23 04:57:26.284483_t=600.pkl",
    # "enumerationResults/propSim_2021-06-10 00:01:40.381552_t=600.pkl",
    # "enumerationResults/uniformGrammar_2021-05-11 13:49:32.922951_t=600.pkl"]
    # modelNames = [
    # # "neuralRecognizer (samples)", "neuralRecognizer (samples)", 
    # "neuralRecognizer",
    # # "neuralRecognizer (enumerated)",
    # # "propSim (samples)", "propSim (samples)", 
    # # "propSim2 (samples)", "propSim2 (samples)", 
    # "propSim2 (handwritten properties)",
    # "propSim2 (sampled properties)",
    # "unifGrammarPrior"]
    # plotFrontiers(filenames, modelNames)
    
    ######################
    # Enumeration Proxy
    ######################
    
    # nSimList = [50]
    # scoreCutoff = 1.0
    # pseudoCounts = 1
    # fileName = "enumerationResults/propSim_2021-05-23 04:57:26.284483_t=600.pkl"
    # frontiers, times = dill.load(open(fileName, "rb"))
    # unsolvedTasks = [f.task for f in frontiers if len(f.entries) == 0]

    # task2FittedGrammars = comparePropSimFittedToRnnEncoded(tasks, frontiers, baseGrammar, sampledFrontiers, propertyFeatureExtractor, featureExtractorArgs, nSimList, scoreCutoff, pseudoCounts, 
    #     weightedSim=weightedSim, compressSimilar=False, weightByPrior=False, taskSpecificInputs=taskSpecificInputs, verbose=verbose)

    # if save:
    #     filename = DATA_DIR + GRAMMARS_DIR + "nSim={}_weightedSim={}_taskSpecificInputs={}_seed={}.pkl".format(nSimList[0], weightedSim, taskSpecificInputs, args["seed"])
    #     dill.dump(task2FittedGrammars, open(filename, "wb"))

    # propertiesPath = DATA_DIR + SAMPLED_PROPERTIES_DIR + propertiesFilename
    # updateSavedPropertiesWithNewCacheTable(properties, propertiesPath)

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
    # Sample properties
     ########################################################################################################

    # grammarName = "base"
    # seed = 1
    # save = True
    # propertyRequest = arrow(tinput, toutput, tbool)
    
    # grammar = getPropertySamplingGrammar(baseGrammar, grammarName, frontiers, pseudoCounts=1, seed=seed)
    # properties = sampleProperties(grammar, train, propertyRequest, featureExtractor, featureExtractorArgs, save)

    ########################################################################################################
    
    # print(frontiers)
    # explorationCompression(baseGrammar, train, testingTasks=test, featureExtractorArgs=featu
