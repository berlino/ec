import random
from collections import defaultdict
import json
import math
import os
import datetime

import dill
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dreamcoder.dreamcoder import explorationCompression, sleep_recognition
from dreamcoder.utilities import eprint, flatten, testTrainSplit
from dreamcoder.grammar import Grammar
from dreamcoder.task import Task
from dreamcoder.type import Context, arrow, tbool, tlist, tint, t0, UnificationFailure
from dreamcoder.recognition import RecognitionModel

# from dreamcoder.domains.list.listPrimitives import basePrimitives, primitives, McCarthyPrimitives, bootstrapTarget_extra, no_length
# from dreamcoder.domains.arc.arcPrimitives2 import _solve6, basePrimitives, pprint, tcolor
from dreamcoder.domains.arc.arcPrimitives import (
    leafPrimitives,
    basePrimitives,
    pprint,
    tcolor,
    tgridin,
    tgridout,
    tdirection,
    Grid,
)
from dreamcoder.domains.arc.arcPrimitives import (
    _solvefcb5c309,
    _solve50cb2852,
    _solve007bbfb7,
    _solve0520fde7,
    _solvec9e6f938,
    _solvef25fbde4,
    _solve97999447,
    _solve72ca375d,
    _solve5521c0d9,
    _solvece4f8723,
)


def retrieveARCJSONTasks(directory, filenames=None):

    # directory = '/Users/theo/Development/program_induction/ec/ARC/data/training'
    trainingData, testingData = [], []

    for filename in os.listdir(directory):
        if ("json" in filename):
            train, test = retrieveARCJSONTask(filename, directory)
            if filenames is not None:
                if filename in filenames:
                    trainingData.append(train)
                    testingData.append(test)
            else:
                trainingData.append(train)
                testingData.append(test)
    return trainingData, testingData


def retrieveARCJSONTask(filename, directory):
    with open(directory + "/" + filename, "r") as f:
        loaded = json.load(f)

    train = Task(
        filename,
        arrow(tgridin, tgridout),
        [
            ((Grid(gridArray=example["input"]),), Grid(gridArray=example["output"]))
            for example in loaded["train"]
        ],
    )
    
    train.specialTask = ('arc', 5)
    test = Task(
        filename,
        arrow(tgridin, tgridout),
        [
            ((Grid(gridArray=example["input"]),), Grid(gridArray=example["output"]))
            for example in loaded["test"]
        ],
    )

    return train, test


def list_options(parser):
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--train-few", default=False, action="store_true")
    parser.add_argument("--firstTimeEnumerationTimeout", type=int, default=3600)

    # parser.add_argument("-i", type=int, default=10)


def check(filename, f, directory):
    train, test = retrieveARCJSONTask(filename, directory=directory)
    print(train)

    for input, output in train.examples:
        input = input[0]
        if f(input) == output:
            print("HIT")
        else:
            print("MISS")
            print("Got")
            f(input).pprint()
            print("Expected")
            output.pprint()

    return


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

def gridToArray(grid):
    temp = np.full((grid.getNumRows(),grid.getNumCols()),None)
    for yPos,xPos in grid.points:
        temp[yPos, xPos] = str(grid.points[(yPos,xPos)])
    return temp

class ArcCNN(nn.Module):
    special = 'arc'
    
    def __init__(self, tasks=[], testingTasks=[], cuda=False, H=64):
        super(ArcCNN, self).__init__()
        self.CUDA = cuda
        self.recomputeTasks = True
        self.outputDimensionality = H

        if cuda:
            self.CUDA=True
            self.cuda()  # I think this should work?

        self.linear = nn.Linear(300,H)

    def forward(self, v, v2=None):

        return F.relu(self.linear(v))

    def featuresOfTask(self, t, t2=None):  # Take a task and returns [features]

        v = None


        for example in t.examples:
            inputGrid, outputGrid = example
            inputGrid = inputGrid[0]
            exampleVector = np.concatenate((np.array(gridToArray(inputGrid)).flatten(), np.array(gridToArray(outputGrid)).flatten()))
            if v is None:
                v = exampleVector
            else:
                v = np.concatenate((v, exampleVector))

        v = v.astype(np.float32)
        pad_amount = 300 - v.shape[0]
        v = nn.functional.pad(torch.from_numpy(v), (0,pad_amount), 'constant', 0)

        return self(v)

    def featuresOfTasks(self, ts, t2=None):  # Take a task and returns [features]
        """Takes the goal first; optionally also takes the current state second"""
        return [self.featuresOfTask(t) for t in ts]

    def taskOfProgram(self, p, t,
                      lenient=False):
        try:
            pl = executeTower(p,0.05)
            if pl is None or (not lenient and len(pl) == 0): return None
            if len(pl) > 100 or towerLength(pl) > 360: return None

            t = SupervisedTower("tower dream", p)
            return t
        except Exception as e:
            return None


def main(args):
    """
    Takes the return value of the `commandlineArguments()` function as input and
    trains/tests the model on manipulating sequences of numbers.
    """
    random.seed(args.pop("random_seed"))
    single_train_task = args.pop("train_few")

    samples = {
        "007bbfb7.json": _solve007bbfb7,
        "c9e6f938.json": _solvec9e6f938,
        "50cb2852.json": lambda grid: _solve50cb2852(grid)(8),
        "fcb5c309.json": _solvefcb5c309,
        "97999447.json": _solve97999447,
        "f25fbde4.json": _solvef25fbde4,
        "72ca375d.json": _solve72ca375d,
        "5521c0d9.json": _solve5521c0d9,
        "ce4f8723.json": _solvece4f8723,
    }

    import os

    directory = (
        "/".join(os.path.abspath(__file__).split("/")[:-4]) + "/arc-data/data/training"
    )
    print(directory)

    # for key in samples.keys():
    #     check(key, lambda x: samples[key](x), directory)


    if single_train_task:
        # trainTasks = retrieveARCJSONTasks(directory, ["913fb3ed.json", "72ca375d.json","f25fbde4.json","fcb5c309.json","ce4f8723.json","0520fde7.json","c9e6f938.json","97999447.json","5521c0d9.json","007bbfb7.json","d037b0a7.json","5117e062.json","4347f46a.json","50cb2852.json","88a10436.json","a5313dff"])
        trainTasks, testTasks = retrieveARCJSONTasks(directory, ['67a3c6ac.json'])
        # Tile tasks
        # trainTasks = retrieveARCJSONTasks(directory, ["97999447.json", "d037b0a7.json", "4347f46a.json", "50cb2852.json"])
    else:
        trainTasks, testTasks = retrieveARCJSONTasks(directory, None)

    baseGrammar = Grammar.uniform(basePrimitives() + leafPrimitives())
    print("base Grammar {}".format(baseGrammar))

    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "experimentOutputs/list/%s" % timestamp
    os.system("mkdir -p %s" % outputDirectory)

    args.update(
        {"outputPrefix": "%s/list" % outputDirectory, "evaluationTimeout": 300,}
    )

    def resume_from_path(resume):
        try:
            resume = int(resume)
            path = checkpointPath(resume)
        except ValueError:
            path = resume
        with open(path, "rb") as handle:
            result = dill.load(handle)
        resume = len(result.grammars) - 1
        eprint("Loaded checkpoint from", path)
        grammar = result.grammars[-1] if result.grammars else grammar
        return result, grammar

    # arcCNN = ArcCNN()
    # train, test = retrieveARCJSONTask('d23f8c26.json', directory)
    # features = arcCNN.featuresOfTask(train)
    # print(arcCNN.forward(features))

    # resumePath = '/Users/theo/Development/program_induction/results/checkpoints/2020-04-13T18:58:48.642078/'
    # pickledFile = 'list_aic=1.0_arity=3_BO=True_CO=True_ES=1_ET=1200_t_zero=3600_HR=0.0_it=2_MF=10_noConsolidation=False_pc=30.0_RT=1800_RR=False_RW=False_solver=ocaml_STM=True_L=1.0_TRR=default_K=2_topkNotMAP=False_graph=True.pickle'
    # result, grammar = resume_from_path(resumePath + pickledFile)
    # recognitionModel = RecognitionModel(ArcCNN(), baseGrammar)
    # sleep_recognition(result, grammar, [], trainTasks, testTasks, result.allFrontiers.values(), featureExtractor=ArcCNN, activation='tanh', CPUs=1, timeout=180, helmholtzFrontiers=[], helmholtzRatio=0, solver='ocaml', enumerationTimeout=10)
    explorationCompression(baseGrammar, trainTasks, testingTasks=testTasks, **args)
