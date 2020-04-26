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
from dreamcoder.program import Program

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

from dreamcoder.domains.arc.arcPrimitives import *

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
    
    def __init__(self, tasks=[], testingTasks=[], cuda=False, H=64, inputDimensions=25):
        super(ArcCNN, self).__init__()
        self.CUDA = cuda
        self.recomputeTasks = True
        self.outputDimensionality = H

        if cuda:
            self.CUDA=True
            self.cuda()  # I think this should work?

        self.linear = nn.Linear(inputDimensions,H)
        # self.hidden = nn.Linear(H, H)

    def forward(self, v, v2=None):

        v = F.relu(self.linear(v))
        return v.view(-1)

    # def featuresOfTask(self, t, t2=None):  # Take a task and returns [features]
    #     v = None
    #     for example in t.examples[-1:]:
    #         inputGrid, outputGrid = example
    #         inputGrid = inputGrid[0]

    #         inputVector = np.array(gridToArray(inputGrid)).flatten().astype(np.float32)
    #         paddedInputVector = nn.functional.pad(torch.from_numpy(inputVector), (0,900 - inputVector.shape[0]), 'constant', 0)

    #         outputVector = np.array(gridToArray(outputGrid)).flatten().astype(np.float32)
    #         paddedOutputVector = nn.functional.pad(torch.from_numpy(outputVector), (900 - outputVector.shape[0],0), 'constant', 0)

    #         exampleVector = torch.cat([paddedInputVector, paddedOutputVector], dim=0)
    #         if v is None:
    #             v = exampleVector
    #         else:
    #             v = torch.cat([v, exampleVector], dim=0)
    #     return self(v)

    def featuresOfTask(self, t):
        v = None
        for example in t.examples[-1:]:
            inputGrid, outputGrid = example
            inputGrid = inputGrid[0]

            inputColors, outputColors = set(inputGrid.points.values()), set(outputGrid.points.values())
            specialColorsInput = inputColors - outputColors
            specialColorsInputVector = [int(i in specialColorsInput) for i in range(10)]
            specialColorsOutput = outputColors - inputColors
            specialColorsOutputVector = [int(i in specialColorsOutput) for i in range(10)]
            changeDimensions = [int((inputGrid.getNumCols() != outputGrid.getNumCols()) or (inputGrid.getNumRows() != outputGrid.getNumRows()))]
            useSplitBlocks = [int(((inputGrid.getNumCols()//outputGrid.getNumCols()) == 2) or ((inputGrid.getNumRows()//outputGrid.getNumRows()) == 2))]
            fractionBlackBInput = [sum([c == 0 for c in inputGrid.points.values()]) / len(inputGrid.points)]
            fractionBlackBOutput = [sum([c == 0 for c in outputGrid.points.values()]) / len(outputGrid.points)]
            pixelWiseError = [0 if (changeDimensions[0] == 1) else (sum([outputGrid.points[key] == outputGrid.points[key] for key in outputGrid.points.keys()]) / len(outputGrid.points))]

            finalVector = np.array([specialColorsInputVector + specialColorsOutputVector + changeDimensions + useSplitBlocks + fractionBlackBInput + fractionBlackBOutput + pixelWiseError]).astype(np.float32)
            finalTensor = torch.from_numpy(finalVector)
            # print(finalTensor)
            if v is None:
                v = finalTensor
            else:
                v = torch.cat([v, finalTensor], dim=0)
        return self(v)


    def featuresOfTasks(self, ts, t2=None):  # Take a task and returns [features]
        """Takes the goal first; optionally also takes the current state second"""
        return [self.featuresOfTask(t) for t in ts]

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

def getTrainFrontier(resumePath, n):
    result, resumeGrammar = resume_from_path(resumePath)
    firstFrontier = [frontier[0] for (key,frontier) in result.frontiersOverTime.items() if len(frontier[0].entries) > 0]
    expandedFrontier = expandFrontier(firstFrontier, n)
    return expandedFrontier, resumeGrammar

def getTask(taskName, allTasks):
    for task in allTasks:
        if task.name == taskName:
            return task
    raise Exception

def scoreProgram(p, recognizer=None, grammar=None, taskName=None, task=None):
    if taskName:
        task = getTask(taskName, trainTasks)
    if recognizer is not None:
        grammar = recognizer.grammarOfTask(task).untorch()
    ll = grammar.logLikelihood(arrow(tgridin, tgridout), p)
    return ll

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
        trainTasks, testTasks = retrieveARCJSONTasks(directory, ['3631a71a.json'])
        # Tile tasks
        # trainTasks = retrieveARCJSONTasks(directory, ["97999447.json", "d037b0a7.json", "4347f46a.json", "50cb2852.json"])
    else:
        trainTasks, testTasks = retrieveARCJSONTasks(directory, None)

    baseGrammar = Grammar.uniform(basePrimitives() + leafPrimitives())
    # print("base Grammar {}".format(baseGrammar))

    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "experimentOutputs/arc/%s" % timestamp
    os.system("mkdir -p %s" % outputDirectory)

    args.update(
        {"outputPrefix": "%s/arc" % outputDirectory, "evaluationTimeout": 1,}
    )

    # # nnTrainTask, _ = retrieveARCJSONTasks(directory, ['dae9d2b5.json'])
    # # arcNN = ArcCNN(inputDimensions=25)
    # # v = arcNN.featuresOfTask(nnTrainTask[0])
    # # print(v)

    resumePath = '/Users/theo/Development/program_induction/results/checkpoints/2020-04-13T18:58:48.642078/'
    pickledFile = 'list_aic=1.0_arity=3_BO=True_CO=True_ES=1_ET=1200_t_zero=3600_HR=0.0_it=2_MF=10_noConsolidation=False_pc=30.0_RT=1800_RR=False_RW=False_solver=ocaml_STM=True_L=1.0_TRR=default_K=2_topkNotMAP=False_graph=True.pickle'
    expandedFrontiers, resumeGrammar = getTrainFrontier(resumePath + pickledFile, 0)
    # # recognitionModel = RecognitionModel(ArcCNN(), baseGrammar)
    # # for frontier in expandedFrontiers:
    # #     task = frontier.task
    # #     print('Task: {}'.format(task.name))
    # #     print(ArcCNN().featuresOfTask(task))
    # # timeout = 1200
    # # path = "recognitionModels/{}_trainTasks={}_timeout={}".format(datetime.datetime.now(), len(expandedFrontiers), timeout)
    # # trainedRecognizer = sleep_recognition(None, baseGrammar, [], [], [], expandedFrontiers, featureExtractor=ArcCNN, activation='tanh', CPUs=1, timeout=timeout, helmholtzFrontiers=[], helmholtzRatio=0, solver='ocaml', enumerationTimeout=0, skipEnumeration=True)
    # # with open(path,'wb') as handle:
    # #     dill.dump(trainedRecognizer, handle)
    # #     print('Stored recognizer at: {}'.format(path))

    # # def getTaskProgram(taskName):

    # #     key = taskName.split('_')[1]
    # #     return taskToProgram[key]

    # # taskToProgram = {frontier.task.name:getTaskProgram(frontier) for frontier in expandedFrontier}

    # # nnTrainTasks = [frontier.task for frontier in expandedFrontier]
    # # for task,program in manuallySolvedTasks.items():
    # #     if task not in taskToProgram:
    # #         taskToProgram[task] = Program.parse(manuallySolvedTasks[task])


    # trainedRecognizerPath = 'recognitionModels/2020-04-26 15:05:28.972185_trainTasks=2343_timeout=1200'
    # with open(trainedRecognizerPath, 'rb') as handle:
    #     trainedRecognizer = dill.load(handle)
    # # print('{} Train Tasks'.format(len(nnTrainTasks)))
    # # scoreTasks(trainedRecognizer, nnTrainTasks, taskToProgram, True)
    # # print('\n {} Test Tasks'.format(len(nnTestTasks)))
    # # scoreTasks(trainedRecognizer, nnTestTasks, taskToProgram, True)

    # print('\n ------------------------------ Train ------------------------------------ \n')

    otherGrammar = baseGrammar.insideOutside(expandedFrontiers, 3)

    # for frontier in expandedFrontiers:
    #     llBefore = scoreProgram(frontier.entries[0].program, grammar=baseGrammar)
    #     llAfter = scoreProgram(frontier.entries[0].program, grammar=otherGrammar)
    #     # llResume = scoreProgram(frontier.entries[0].program, grammar=resumeGrammar)
    #     # llAfter = scoreProgram(frontier.entries[0].program, trainedRecognizer, task=frontier.task)
    #     # print("{}".format(Grammar.uniform(basePrimitives() + leafPrimitives())))
    #     print("{}: {}, {}".format(frontier.task.name, llBefore, llAfter))
    #     print("Program: {}".format(frontier.entries[0].program))

    # print('\n ------------------------------ Test ------------------------------------ \n')        

    # for task,program in manuallySolvedTasks.items():
    #     if task not in [frontier.task.name for frontier in expandedFrontiers]:
    #         p = Program.parse(manuallySolvedTasks[task])
    #         llBefore = scoreProgram(p, grammar=baseGrammar)
    #         llAfter = scoreProgram(p, grammar=otherGrammar)
    #         # try:
    #         #     llResume = scoreProgram(p, grammar=resumeGrammar)
    #         # except:
    #         #     llResume = -1
    #         # llAfter = scoreProgram(p, trainedRecognizer, task=frontier.task)
    #         print("{}: {}, {}".format(frontier.task.name, llBefore, llAfter))
    #         print("Program: {}".format(p))

    explorationCompression(otherGrammar, trainTasks, testingTasks=testTasks, **args)
