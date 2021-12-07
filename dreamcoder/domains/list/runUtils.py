import torch.nn as nn
import torch

from dreamcoder.domains.list.handwrittenProperties import handWrittenProperties, getHandwrittenPropertiesFromTemplates, tinput, toutput
from dreamcoder.domains.list.listPrimitives import basePrimitives, primitives, McCarthyPrimitives, bootstrapTarget_extra, no_length, josh_primitives
from dreamcoder.domains.list.makeListTasks import make_list_bootstrap_tasks, sortBootstrap, EASYLISTTASKS, joshTasks
from dreamcoder.domains.list.propertySignatureExtractor import PropertySignatureExtractor
from dreamcoder.recognition import DummyFeatureExtractor, RecognitionModel
from dreamcoder.task import Task
from dreamcoder.type import Context, arrow, tbool, tlist, tint, t0, UnificationFailure
from dreamcoder.utilities import flatten, numberOfCPUs

DATA_DIR = "data/prop_sig/"
SAMPLED_PROPERTIES_DIR = "sampled_properties/"
GRAMMARS_DIR = "grammars/"

def list_options(parser):

    # parser.add_argument("--iterations", type=int, default=10)
    # parser.add_argument("--useDSL", action="store_true", default=False)
    parser.add_argument("--libraryName",  default="property_prims", choices=[
        "josh_1",
        "josh_2",
        "josh_3",
        "josh_3.1",
        "josh_final",
        "josh_rich_0_10",
        "josh_rich_0_99",
        "property_prims",
        "dc_list_domain"])
    parser.add_argument("--propSamplingPrimitives", default="same", choices=[
        "same",
        "josh_1",
        "josh_2",
        "josh_3",
        "josh_3.1",
        "josh_final",
        "josh_rich_0_10",
        "josh_rich_0_99",
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
            "josh_3_long_inputs_0_10",
            "josh_3.1",
            "josh_final",
            "josh_fleet_0_99",
            "josh_fleet_10_99",
            "josh_fleet_0_10",
            "Lucas-old"])
    parser.add_argument("--extractor", default="prop_sig", choices=[
        "prop_sig",
        "learned",
        "combined",
        "dummy"
        ])
    parser.add_argument("--hidden", type=int, default=64)

    # Arguments relating to propSim
    parser.add_argument("--enumerationProxy", action="store_true", default=False)
    parser.add_argument("--propSim", action="store_true", default=False)
    parser.add_argument("--helmEnumerationTimeout", type=int, default=1)
    parser.add_argument("--propNumIters", type=int, default=1)
    parser.add_argument("--hmfSeed", type=int, default=None)
    parser.add_argument("--numHelmFrontiers", type=int, default=None)
    parser.add_argument("--maxFractionSame", type=float, default=1.0)
    parser.add_argument("--helmholtzFrontiers", type=str, default=None)
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

def get_tasks(dataset):
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
        "josh_3_long_inputs_0_10": lambda: joshTasks("3_long_inputs_0_10"),
        "josh_3.1": lambda: joshTasks("3.1"),
        "josh_final": lambda: joshTasks("final"),
        "josh_fleet_0_99": lambda: joshTasks("fleet_0_99"),
        "josh_fleet_0_10": lambda: joshTasks("fleet_0_10"),
        "josh_fleet_10_99": lambda: joshTasks("fleet_10_99")
    }[dataset]()

    return tasks

def get_primitives(libraryName):
    primLibraries = {
             "josh_1": josh_primitives("1"),
             "josh_2": josh_primitives("2"),
             "josh_3": josh_primitives("3")[0],
             "josh_3.1": josh_primitives("3.1")[0],
             "josh_final": josh_primitives("final"),
             "josh_rich_0_10": josh_primitives("rich_0_10"),
             "josh_rich_0_99": josh_primitives("rich_0_99"),
             "property_prims": handWrittenProperties(),
             "dc_list_domain": bootstrapTarget_extra()
    }
    prims = primLibraries[libraryName]
    return prims

def get_extractor(tasks, baseGrammar, args):
    extractor = {
        "dummy": DummyFeatureExtractor,
        "learned": LearnedFeatureExtractor,
        "prop_sig": PropertySignatureExtractor,
        "combined": CombinedExtractor
        }[args["extractor"]]

    if args["extractor"] == "learned":
        raise NotImplementedError

    elif args["extractor"] == "prop_sig" or extractorName == "combined":
        featureExtractorArgs = args

        if args["propToUse"] == "handwritten":
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
                    for task in tasksToSolve:
                        allProperties[task] = allProperties.get(task, []) + featureExtractor.properties[task]
                # assertion triggered if 0 properties enumerated
                except AssertionError:
                    print("0 properties found")
            
            for task in tasksToSolve:
                print("Found {} properties for task {}".format(len(allProperties.get(task, [])), task))
                for p in sorted(allProperties.get(task, []), key=lambda p: p.score, reverse=True):
                    print("program: {} \nreturnType: {} \nprior: {:.2f} \nscore: {:.2f}".format(p, p.request.returns(), p.logPrior, p.score))
                    print("-------------------------------------------------------------")

            if save:
                filename = "sampled_properties_weights={}_sampling_timeout={}s_return_types={}_seed={}.pkl".format(
                    propSamplingGrammarWeights, int(featureExtractorArgs["propEnumerationTimeout"]), returnTypes, args["seed"])
                savePath = DATA_DIR + SAMPLED_PROPERTIES_DIR + filename
                dill.dump(allProperties, open(savePath, "wb"))
                print("Saving sampled properties at: {}".format(savePath))
                print("Saving sampled properties at: {}".format(savePath))

        return featureExtractor, properties
