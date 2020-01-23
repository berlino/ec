import random
from collections import defaultdict
import json
import math
import os
import datetime

from dreamcoder.dreamcoder import explorationCompression
from dreamcoder.utilities import eprint, flatten, testTrainSplit
from dreamcoder.grammar import Grammar
from dreamcoder.task import Task
from dreamcoder.type import Context, arrow, tbool, tlist, tint, t0, UnificationFailure
# from dreamcoder.domains.list.listPrimitives import basePrimitives, primitives, McCarthyPrimitives, bootstrapTarget_extra, no_length
# from dreamcoder.domains.arc.arcPrimitives2 import _solve6, basePrimitives, pprint, tcolor
from dreamcoder.domains.arc.arcPrimitives import leafPrimitives, basePrimitives, pprint, tcolor, tgrid, tgrids, tdirection, Grid
from dreamcoder.domains.arc.arcPrimitives import \
    _solvefcb5c309,\
    _solve50cb2852, \
    _solve007bbfb7,\
    _solve0520fde7, \
    _solvec9e6f938,\
    _solvef25fbde4, \
    _solve97999447, \
    _solve72ca375d, \
    _solve5521c0d9, \
    _solvece4f8723

from dreamcoder.recognition import RecurrentFeatureExtractor
from dreamcoder.domains.list.makeListTasks import make_list_bootstrap_tasks, sortBootstrap, EASYLISTTASKS


# def retrieveJSONTasks(filename, features=False):
#     """
#     For JSON of the form:
#         {"name": str,
#          "type": {"input" : bool|int|list-of-bool|list-of-int,
#                   "output": bool|int|list-of-bool|list-of-int},
#          "examples": [{"i": data, "o": data}]}
#     """
#     with open(filename, "r") as f:
#         loaded = json.load(f)
#     TP = {
#         "bool": tbool,
#         "int": tint,
#         "list-of-bool": tlist(tbool),
#         "list-of-int": tlist(tint),
#     }
#     return [Task(
#         item["name"],
#         arrow(TP[item["type"]["input"]], TP[item["type"]["output"]]),
#         [((ex["i"],), ex["o"]) for ex in item["examples"]],
#         features=(None if not features else list_features(
#             [((ex["i"],), ex["o"]) for ex in item["examples"]])),
#         cache=False,
#     ) for item in loaded]

def retrieveARCJSONTasks(directory, filenames=None):

    # directory = '/Users/theo/Development/program_induction/ec/ARC/data/training'
    trainingData = []

    for filename in os.listdir(directory):
        train, test = retrieveARCJSONTask(filename, directory)
        if filenames is not None:
            if filename in filenames:
                trainingData.append(train)
        else:
            trainingData.append(train)
    return trainingData


def retrieveARCJSONTask(filename, directory):
    with open(directory + '/' + filename, "r") as f:
        loaded = json.load(f)

    train = Task(filename, arrow(tgrid, tgrid), [((Grid(gridArray=example['input']),), Grid(gridArray=example['output'])) for example in loaded['train']])
    test = Task(filename, arrow(tgrid, tgrid), [((Grid(gridArray=example['input']),), Grid(gridArray=example['output'])) for example in loaded['test']])

    return train, test

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


# class LearnedFeatureExtractor(RecurrentFeatureExtractor):
#     H = 64
#
#     special = None
#
#     def tokenize(self, examples):
#         def sanitize(l): return [z if z in self.lexicon else "?"
#                                  for z_ in l
#                                  for z in (z_ if isinstance(z_, list) else [z_])]
#
#         tokenized = []
#         for xs, y in examples:
#             if isinstance(y, list):
#                 y = ["LIST_START"] + y + ["LIST_END"]
#             else:
#                 y = [y]
#             y = sanitize(y)
#             if len(y) > self.maximumLength:
#                 return None
#
#             serializedInputs = []
#             for xi, x in enumerate(xs):
#                 if isinstance(x, list):
#                     x = ["LIST_START"] + x + ["LIST_END"]
#                 else:
#                     x = [x]
#                 x = sanitize(x)
#                 if len(x) > self.maximumLength:
#                     return None
#                 serializedInputs.append(x)
#
#             tokenized.append((tuple(serializedInputs), y))
#
#         return tokenized
#
#     def __init__(self, tasks, testingTasks=[], cuda=False):
#         self.lexicon = set(flatten((t.examples for t in tasks + testingTasks), abort=lambda x: isinstance(
#             x, str))).union({"LIST_START", "LIST_END", "?"})
#
#         # Calculate the maximum length
#         self.maximumLength = float('inf') # Believe it or not this is actually important to have here
#         self.maximumLength = max(len(l)
#                                  for t in tasks + testingTasks
#                                  for xs, y in self.tokenize(t.examples)
#                                  for l in [y] + [x for x in xs])
#
#         self.recomputeTasks = True
#
#         super(
#             LearnedFeatureExtractor,
#             self).__init__(
#             lexicon=list(
#                 self.lexicon),
#             tasks=tasks,
#             cuda=cuda,
#             H=self.H,
#             bidirectional=True)


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
    parser.add_argument("--random-seed", type=int, default=17)
    # parser.add_argument("-i", type=int, default=10)

def check(filename, f, directory):
    train, test = retrieveARCJSONTask(filename, directory=directory)
    print(train)

    for input, output in train.examples:
        input = input[0]
        if f(input) == output:
            print('HIT')
        else:
            print('MISS')
            print('Got')
            f(input).pprint()
            print('Expected')
            output.pprint()

    return

def main(args):
    """
    Takes the return value of the `commandlineArguments()` function as input and
    trains/tests the model on manipulating sequences of numbers.
    """
    random.seed(args.pop("random_seed"))

    samples = {
        '007bbfb7.json': _solve007bbfb7,
        'c9e6f938.json': _solvec9e6f938,
        '50cb2852.json': lambda grid: _solve50cb2852(grid)(8),
        'fcb5c309.json': _solvefcb5c309,
        '97999447.json': _solve97999447,
        'f25fbde4.json': _solvef25fbde4,
        '72ca375d.json': _solve72ca375d,
        '5521c0d9.json': _solve5521c0d9,
        'ce4f8723.json': _solvece4f8723
    }


    import os

    directory = '/'.join(os.path.abspath(__file__).split('/')[:-4]) + '/arc-data/data/training'
    print(directory)

    for key in samples.keys():
        check(key, lambda x: samples[key](x), directory)

    trainTasks = retrieveARCJSONTasks(directory, None)

    baseGrammar = Grammar.uniform(basePrimitives() + leafPrimitives())
    print('base Grammar {}'.format(baseGrammar))

    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "experimentOutputs/list/%s"%timestamp
    os.system("mkdir -p %s"%outputDirectory)
    
    args.update({
        "outputPrefix": "%s/list"%outputDirectory,
        "evaluationTimeout": 300,
    })


    # #
    request = arrow(tgrid, tgrid)
    #
    # for ll,_,p in baseGrammar.enumeration(Context.EMPTY, [], request, 13):
    #     ll_ = baseGrammar.logLikelihood(request,p)
    #     print(ll, p, ll_)

    # baseGrammar = Grammar.uniform(basePrimitives())
    # print(baseGrammar.buildCandidates(request, Context.EMPTY, [], returnTable=True))

    explorationCompression(baseGrammar, trainTasks, testingTasks=[], **args)