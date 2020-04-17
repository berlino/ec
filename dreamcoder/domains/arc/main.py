import random
from collections import defaultdict
import json
import math
import os
import datetime

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dreamcoder.dreamcoder import explorationCompression
from dreamcoder.utilities import eprint, flatten, testTrainSplit
from dreamcoder.grammar import Grammar
from dreamcoder.task import Task
from dreamcoder.type import Context, arrow, tbool, tlist, tint, t0, UnificationFailure

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

from dreamcoder.recognition import RecurrentFeatureExtractor
from dreamcoder.domains.list.makeListTasks import (
    make_list_bootstrap_tasks,
    sortBootstrap,
    EASYLISTTASKS,
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

class ArcCNN(nn.Module):
    special = 'arc'
    
    def __init__(self, tasks, testingTasks=[], cuda=False, H=64):
        super(ArcCNN, self).__init__()
        self.CUDA = cuda
        self.recomputeTasks = True

        self.outputDimensionality = H
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                # nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

        self.inputImageDimension = 256
        self.resizedDimension = 64
        assert self.inputImageDimension % self.resizedDimension == 0

        # channels for hidden
        hid_dim = 64
        z_dim = 64

        self.encoder = nn.Sequential(
            conv_block(6, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
            Flatten()
        )

        self.outputDimensionality = 1024

        if cuda:
            self.CUDA=True
            self.cuda()  # I think this should work?

    def forward(self, v, v2=None):
        """v: tower to build. v2: image of tower we have built so far"""
        # insert batch if it is not already there
        # if len(v.shape) == 3:
        #     v = np.expand_dims(v, 0)
        #     inserted_batch = True
        #     if v2 is not None:
        #         assert len(v2.shape) == 3
        #         v2 = np.expand_dims(v2, 0)
        # elif len(v.shape) == 4:
        #     inserted_batch = False
        #     pass
        # else:
        #     assert False, "v has the shape %s"%(str(v.shape))
        
        # if v2 is None: v2 = np.zeros(v.shape)
        
        # v = np.concatenate((v,v2), axis=3)
        # v = np.transpose(v,(0,3,1,2))
        # assert v.shape == (v.shape[0], 6,self.inputImageDimension,self.inputImageDimension)
        # v = variable(v, cuda=self.CUDA).float()
        # window = int(self.inputImageDimension/self.resizedDimension)
        # v = F.avg_pool2d(v, (window,window))
        # #showArrayAsImage(np.transpose(v.data.numpy()[0,:3,:,:],[1,2,0]))
        # v = self.encoder(v)
        # if inserted_batch:
        #     return v.view(-1)
        # else:
        return F.relu(x)

    def featuresOfTask(self, t, t2=None):  # Take a task and returns [features]

        v = None
        for example in t.examples:
            inputGrid, outputGrid = example
            inputGrid = inputGrid[0]
            exampleVector = np.concatenate((np.array(inputGrid.gridArray).flatten(), np.array(outputGrid.gridArray).flatten()))
            if v is None:
                v = exampleVector
            else:
                v = np.concatenate((v, exampleVector))
        print(v)
        return v
    
    def featuresOfTasks(self, ts, t2=None):  # Take a task and returns [features]
        """Takes the goal first; optionally also takes the current state second"""
        if t2 is None:
            pass
        elif isinstance(t2, Task):
            assert False
            #t2 = np.array([t2.getImage(drawHand=True)]*len(ts))
        elif isinstance(t2, list):
            t2 = np.array([t.getImage(drawHand=True) if t else np.zeros((self.inputImageDimension,
                                                                         self.inputImageDimension,
                                                                         3))
                           for t in t2])
        else:
            assert False
            
        return self(np.array([t.getImage() for t in ts]),
                    t2)

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

    for key in samples.keys():
        check(key, lambda x: samples[key](x), directory)


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

    # #
    # request = arrow(tgrid, tgrid)
    #
    # for ll,_,p in baseGrammar.enumeration(Context.EMPTY, [], request, 13):
    #     ll_ = baseGrammar.logLikelihood(request,p)
    #     print(ll, p, ll_)

    # baseGrammar = Grammar.uniform(basePrimitives())
    # print(baseGrammar.buildCandidates(request, Context.EMPTY, [], returnTable=True))

    explorationCompression(baseGrammar, trainTasks, testingTasks=testTasks, **args)
