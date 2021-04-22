import random
from collections import defaultdict
import json
import math
import os
import torch.nn as nn
import torch
import datetime

from dreamcoder.dreamcoder import explorationCompression
from dreamcoder.utilities import eprint, flatten, testTrainSplit, numberOfCPUs
from dreamcoder.recognition import DummyFeatureExtractor
from dreamcoder.grammar import Grammar
from dreamcoder.task import Task
from dreamcoder.type import Context, arrow, tbool, tlist, tint, t0, UnificationFailure
from dreamcoder.domains.list.listPrimitives import basePrimitives, primitives, McCarthyPrimitives, bootstrapTarget_extra, no_length, josh_primitives
from dreamcoder.domains.list.makeListTasks import make_list_bootstrap_tasks, sortBootstrap, EASYLISTTASKS, joshTasks
from dreamcoder.domains.list.propertySignatureExtractor import PropertySignatureExtractor, sampleProperties
from dreamcoder.domains.list.taskProperties import handWrittenProperties, tinput, toutput

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
        useEmbeddings=True,
        # What should be the timeout for trying to construct Helmholtz tasks?
        helmholtzTimeout=0.25,
        # What should be the timeout for running a Helmholtz program?
        helmholtzEvaluationTimeout=0.01,
        grammar=None,
        featureExtractorArgs=None):
        super(CombinedExtractor, self).__init__()

        self.propSigExtractor = PropertySignatureExtractor(tasks=tasks, testingTasks=testingTasks, H=H, embedSize=embedSize, useEmbeddings=useEmbeddings, helmholtzTimeout=helmholtzTimeout, helmholtzEvaluationTimeout=helmholtzEvaluationTimeout,
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
        "property_prims",
        "list_prims"])
    parser.add_argument("--propSamplingGrammar", default="same", choices=[
        "same"
        "josh_1",
        "josh_2",
        "josh_3",
        "josh_3.1",
        "josh_final",
        "property_prims",
        "list_prims"])
    parser.add_argument(
        "--dataset",
        type=str,
        default="Lucas-old",
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
    parser.add_argument("--singleTask", action="store_true", default=False)
    parser.add_argument("--propCPUs", type=int, default=numberOfCPUs())
    parser.add_argument("--propSolver",default="ocaml",type=str)
    parser.add_argument("--propSamplingTimeout",default=600,type=float)
    parser.add_argument("--propUseConjunction", action="store_true", default=False)
    parser.add_argument("--propAddZeroToNinePrims", action="store_true", default=False)
    parser.add_argument("--propSamplingMethod", default="unique_task_signature", choices=[
        "per_task_discrimination",
        "unique_task_signature"
        ])
    parser.add_argument("--propDreamTasks", action="store_true", default=False)
    parser.add_argument("--propUseHandWrittenProperties", action="store_true", default=False)


def main(args):
    """
    Takes the return value of the `commandlineArguments()` function as input and
    trains/tests the model on manipulating sequences of numbers.
    """

    dataset = args.pop("dataset")
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
        "josh_final": lambda: joshTasks("final")
    }[dataset]()

    # maxTasks = args.pop("maxTasks")
    # if maxTasks and len(tasks) > maxTasks:
    #     necessaryTasks = []  # maxTasks will not consider these
    #     if dataset.startswith("Lucas2.0") and dataset != "Lucas2.0-depth1":
    #         necessaryTasks = tasks[:105]

    #     eprint("Unwilling to handle {} tasks, truncating..".format(len(tasks)))
    #     random.shuffle(tasks)
    #     del tasks[maxTasks:]
    #     tasks = necessaryTasks + tasks

    # if dataset.startswith("Lucas"):
    #     # extra tasks for filter
    #     tasks.extend([
    #         Task("remove empty lists",
    #              arrow(tlist(tlist(tbool)), tlist(tlist(tbool))),
    #              [((ls,), list(filter(lambda l: len(l) > 0, ls)))
    #               for _ in range(15)
    #               for ls in [[[random.random() < 0.5 for _ in range(random.randint(0, 3))]
    #                           for _ in range(4)]]]),
    #         Task("keep squares",
    #              arrow(tlist(tint), tlist(tint)),
    #              [((xs,), list(filter(lambda x: int(math.sqrt(x)) ** 2 == x,
    #                                   xs)))
    #               for _ in range(15)
    #               for xs in [[random.choice([0, 1, 4, 9, 16, 25])
    #                           if random.random() < 0.5
    #                           else random.randint(0, 9)
    #                           for _ in range(7)]]]),
    #         Task("keep primes",
    #              arrow(tlist(tint), tlist(tint)),
    #              [((xs,), list(filter(lambda x: x in {2, 3, 5, 7, 11, 13, 17,
    #                                                   19, 23, 29, 31, 37}, xs)))
    #               for _ in range(15)
    #               for xs in [[random.choice([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37])
    #                           if random.random() < 0.5
    #                           else random.randint(0, 9)
    #                           for _ in range(7)]]]),
    #     ])
    #     for i in range(4):
    #         tasks.extend([
    #             Task("keep eq %s" % i,
    #                  arrow(tlist(tint), tlist(tint)),
    #                  [((xs,), list(filter(lambda x: x == i, xs)))
    #                   for _ in range(15)
    #                   for xs in [[random.randint(0, 6) for _ in range(5)]]]),
    #             Task("remove eq %s" % i,
    #                  arrow(tlist(tint), tlist(tint)),
    #                  [((xs,), list(filter(lambda x: x != i, xs)))
    #                   for _ in range(15)
    #                   for xs in [[random.randint(0, 6) for _ in range(5)]]]),
    #             Task("keep gt %s" % i,
    #                  arrow(tlist(tint), tlist(tint)),
    #                  [((xs,), list(filter(lambda x: x > i, xs)))
    #                   for _ in range(15)
    #                   for xs in [[random.randint(0, 6) for _ in range(5)]]]),
    #             Task("remove gt %s" % i,
    #                  arrow(tlist(tint), tlist(tint)),
    #                  [((xs,), list(filter(lambda x: not x > i, xs)))
    #                   for _ in range(15)
    #                   for xs in [[random.randint(0, 6) for _ in range(5)]]])
    #         ])

    # def isIdentityTask(t):
    #     return all( len(xs) == 1 and xs[0] == y for xs, y in t.examples  )
    # eprint("Removed", sum(isIdentityTask(t) for t in tasks), "tasks that were just the identity function")
    # tasks = [t for t in tasks if not isIdentityTask(t) ]

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
             "property_prims": handWrittenProperties(),
             "list_prims": bootstrapTarget_extra()
    }

    prims = primLibraries[args.pop("primitives")]

    haveLength = not args.pop("noLength")
    haveMap = not args.pop("noMap")
    haveUnfold = not args.pop("noUnfold")
    eprint(f"Including map as a primitive? {haveMap}")
    eprint(f"Including length as a primitive? {haveLength}")
    eprint(f"Including unfold as a primitive? {haveUnfold}")

    if "josh" in dataset:
        tasks = [t for t in tasks if int(t.name[:3]) < 81 and "_1" in t.name]
    tasks = [t for t in tasks if (t.request == arrow(tlist(tint), tlist(tint)) and isinstance(t.examples[0][1],list) and isinstance(t.examples[0][0][0],list))]

    # for t in tasks:
    #     t.request = arrow(tinput, toutput)

    if tasks[0].request == tinput and "tinput_to_tlist" not in [p.name for p in prims]:
        toutputToList = Primitive("toutput_to_tlist", arrow(toutput, tlist(tint)), lambda x: x)
        tinputToList = Primitive("tinput_to_tlist", arrow(tinput, tlist(tint)), lambda x: x)
        prims = prims + [toutputToList, tinputToList]

    baseGrammar = Grammar.uniform([p
                                   for p in prims
                                   if (p.name != "map" or haveMap) and \
                                   (p.name != "unfold" or haveUnfold) and \
                                   (p.name != "length" or haveLength)])

    extractor_name = args.pop("extractor")
    print(extractor_name)
    extractor = {
        "dummy": DummyFeatureExtractor,
        "learned": LearnedFeatureExtractor,
        "prop_sig": PropertySignatureExtractor,
        "combined": CombinedExtractor
        }[extractor_name]



    hidden = args.pop("hidden")
    propCPUs = args.pop("propCPUs")
    propSolver = args.pop("propSolver")
    propSamplingTimeout = args.pop("propSamplingTimeout")
    propUseConjunction = args.pop("propUseConjunction")
    propAddZeroToNinePrims = args.pop("propAddZeroToNinePrims")
    propSamplingMethod = args.pop("propSamplingMethod")
    propDreamTasks = args.pop("propDreamTasks")
    propUseHandWrittenProperties = args.pop("propUseHandWrittenProperties")
    propSamplingGrammar = args.pop("propSamplingGrammar")

    if extractor_name == "learned":
        featureExtractorArgs = {"hidden":hidden}
    elif extractor_name == "prop_sig" or extractor_name == "combined":
        featureExtractorArgs = {
            "propCPUs": propCPUs,
            "propSolver": propSolver,
            "propSamplingTimeout": propSamplingTimeout,
            "propUseConjunction": propUseConjunction,
            "propAddZeroToNinePrims": propAddZeroToNinePrims,
            "propSamplingMethod": propSamplingMethod,
            "propDreamTasks": propDreamTasks,
            "propUseHandWrittenProperties": propUseHandWrittenProperties,
            "propSamplingGrammar": propSamplingGrammar,
            "primLibraries": primLibraries
        }

    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "experimentOutputs/jrule/%s"%timestamp

    os.system("mkdir -p %s"%outputDirectory)
    
    args.update({
        "featureExtractor": extractor,
        "outputPrefix": "%s/jrule"%outputDirectory,
        "evaluationTimeout": 0.0005,
    })

    eprint("Got {} list tasks".format(len(tasks)))
    split = args.pop("split")

    if split:
        train_some = defaultdict(list)
        for t in tasks:
            necessary = train_necessary(t)
            if not necessary:
                continue
            if necessary == "some":
                train_some[t.name.split()[0]].append(t)
            else:
                t.mustTrain = True
        for k in sorted(train_some):
            ts = train_some[k]
            random.shuffle(ts)
            ts.pop().mustTrain = True

        test, train = testTrainSplit(tasks, split)
        if True:
            test = [t for t in test
                    if t.name not in EASYLISTTASKS]

        eprint(
            "Alotted {} tasks for training and {} for testing".format(
                len(train), len(test)))
    else:
        train = tasks
        test = []

    singleTask = args.pop("singleTask")
    if singleTask:
        train = [train[0]]

    explorationCompression(baseGrammar, train, testingTasks=test, featureExtractorArgs=featureExtractorArgs, **args)
