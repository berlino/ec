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
from dreamcoder.utilities import eprint, flatten, testTrainSplit, lse
from dreamcoder.grammar import Grammar, ContextualGrammar
from dreamcoder.task import Task
from dreamcoder.type import Context, arrow, tbool, tlist, tint, t0, UnificationFailure
from dreamcoder.recognition import RecognitionModel
from dreamcoder.program import Program
from dreamcoder.domains.arc.arcPrimitives import *
from dreamcoder.domains.arc.taskGeneration import *

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
    return result, grammar, result.grammars[0]

def getLearnedProductions(result):
    currentProductions = set()
    learnedProductions = {}
    for i,grammar in enumerate(result.grammars):
        learnedProductions[i] = {}
        productions = set([str(p[2]) for p in grammar.productions])
        if len(list(currentProductions)) == 0:
            currentProductions = productions
        else:
            print('----------------------------------------------------{}-----------------------------------------------------'.format(i))        
            newProductions = productions.difference(currentProductions)
            for j,production in enumerate(newProductions):
                learnedProductions[i][j] = production
                print(j, production)
        currentProductions = currentProductions.union(productions)
    return learnedProductions


def getTrainFrontier(resumePath, n):
    result, resumeGrammar, preConsolidationGrammar = resume_from_path(resumePath)
    firstFrontier = [frontiers[0] for (key,frontiers) in result.frontiersOverTime.items() if len(frontiers[0].entries) > 0]
    allFrontiers = [frontier for (task,frontier) in result.allFrontiers.items() if len(frontier.entries) > 0]
    # expandedFrontier = expandFrontier(firstFrontier, n)
    print(result.learningCurve)
    learnedProductions = getLearnedProductions(result)
    return firstFrontier, allFrontiers, result.frontiersOverTime, resumeGrammar, preConsolidationGrammar, result.recognitionModel, learnedProductions

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
    if grammar is None:
        return 0
    ll = grammar.logLikelihood(arrow(tgridin, tgridout), p)
    return ll

def normalizeProductions(grammar):
    z = lse([l for l,_,p in grammar.productions])
    normalizedProductions = [(p[0]-z, p[2]) for p in grammar.productions]
    return Grammar.fromProductions(normalizedProductions)

def upweightProduction(name, scaleFactor, grammar):
    productions = [(p[0],p[2]) for p in grammar.productions]
    for i in range(len(productions)):
        if str(productions[i][1]) == name:
            print('production before: {}'.format(productions[i][0]))
            productions[i] = (productions[i][0] + math.log(scaleFactor), productions[i][1])
            print('production after: {}'.format(productions[i][0]))
    return Grammar.fromProductions(productions)

def upweightConditionalProduction(parentPrimitive, argumentIndex, production, scaleFactor, contextualGrammar):
    primitiveGrammar = contextualGrammar.library[parentPrimitive][argumentIndex]
    print('primitiveGrammar before: ', primitiveGrammar)
    newPrimitiveGrammar = upweightProduction(production, scaleFactor, primitiveGrammar)
    print('conditionalGrammar after: ', newPrimitiveGrammar)
    newContextualGrammar = copy.deepcopy(contextualGrammar)
    newContextualGrammar.library[parentPrimitive][argumentIndex] = newPrimitiveGrammar
    return newContextualGrammar

def upweightConditionalProductions(parent2UpdateProduction, scaleFactor, contextualGrammar):
    currentGrammar = contextualGrammar
    for ((parentPrimitive, argIndex),production) in parent2UpdateProduction.items():
        currentGrammar = upweightConditionalProduction(parentPrimitive, argIndex, production, scaleFactor, currentGrammar)
    return currentGrammar

def evaluateGrammars(dreamcoderFrontier, manuallySolvedTasks, grammar1=None, grammar2=None, recognizer1=None, recognizer2=None):

    print('\n ------------------------------ Solved by Dreamcoder ------------------------------------ \n')
    averageLLBefore, averageLLAfter = 0,0
    for frontier in dreamcoderFrontier:
        llBefore = scoreProgram(frontier.entries[0].program, grammar=grammar1, recognizer=recognizer1)
        llAfter = scoreProgram(frontier.entries[0].program, grammar=grammar2, recognizer=recognizer2)
        if llAfter == 0:
            print("{}: {}".format(frontier.task, llBefore))
        else:
            print("{}: {} -> {}".format(frontier.task.name, llBefore, llAfter))
        print("Program: {}\n".format(frontier.entries[0].program))
        averageLLBefore += llBefore
        averageLLAfter += llAfter

    print("Solved {} tasks".format(len(dreamcoderFrontier)))
    if llAfter == 0:
        print('Average LL: {}'.format(averageLLBefore / len(dreamcoderFrontier)))
    else:
        print('Average LL: {} -> {}'.format(averageLLBefore / len(dreamcoderFrontier), (averageLLAfter / len(dreamcoderFrontier))))

    print('\n ------------------------------ Manually Solved ------------------------------------ \n')        


    solvedByDreamcoder = 0
    averageLLBefore, averageLLAfter = 0,0
    for task,program in manuallySolvedTasks.items():
        if task not in [frontier.task.name for frontier in dreamcoderFrontier]:
            p = Program.parse(manuallySolvedTasks[task])
            llBefore = scoreProgram(p, grammar=grammar1, recognizer=recognizer1)
            llAfter = scoreProgram(p, grammar=grammar2, recognizer=recognizer2)
            if llAfter == 0:
                print("{}: {}".format(task, llBefore))
            else:
                print("{}: {} -> {}".format(task, llBefore, llAfter))
            print("Program: {}\n".format(p))
            averageLLBefore += llBefore
            averageLLAfter += llAfter
        else:
            solvedByDreamcoder += 1
    
    print("{} of {} manually written solutions were found by Dreamcoder".format(solvedByDreamcoder, len(manuallySolvedTasks)))
    numUnsolved = len(manuallySolvedTasks) - solvedByDreamcoder
    if llAfter == 0:
        print('Average LL: {}'.format(averageLLBefore / numUnsolved))
    else:
        print('Average LL: {} -> {}'.format(averageLLBefore/numUnsolved, averageLLAfter/numUnsolved))


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

    directory = "/".join(os.path.abspath(__file__).split("/")[:-4]) + "/arc-data/data/"

    if single_train_task:
        # trainTasks = retrieveARCJSONTasks(directory, ["913fb3ed.json", "72ca375d.json","f25fbde4.json","fcb5c309.json","ce4f8723.json","0520fde7.json","c9e6f938.json","97999447.json","5521c0d9.json","007bbfb7.json","d037b0a7.json","5117e062.json","4347f46a.json","50cb2852.json","88a10436.json","a5313dff"])
        trainTasks, testTasks = retrieveARCJSONTasks(directory, ['3631a71a.json'])
        # Tile tasks
        # trainTasks = retrieveARCJSONTasks(directory, ["97999447.json", "d037b0a7.json", "4347f46a.json", "50cb2852.json"])
    else:
        trainTasks, _ = retrieveARCJSONTasks(directory + 'training', None)

    testTasks, _ = retrieveARCJSONTasks(directory + 'evaluation')

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

    resumePath = '/Users/theo/Development/program_induction/experimentOutputs/arc/'
    resumeDirectory = '2020-05-10T14:49:21.186479/'
    pickledFile = 'arc_aic=1.0_arity=3_BO=True_CO=True_ES=1_ET=1200_t_zero=28800_HR=0.0_it=6_MF=10_noConsolidation=False_pc=1.0_RT=1800_RR=False_RW=False_solver=ocaml_STM=True_L=1.0_TRR=unsolved_K=2_topkNotMAP=False.pickle'
    firstFrontier, allFrontiers, frontierOverTime, topDownGrammar, preConsolidationGrammar, resumeRecognizer, learnedProductions = getTrainFrontier(resumePath + resumeDirectory + pickledFile, 0)

    def convertFrontiersOverTimeToJson(frontiersOverTime):
        frontiersOverTimeJson = {}
        numFrontiers = len(list(frontiersOverTime.values())[0])
        # print('{} frontiers per task'.format(numFrontiers))
        for task,frontiers in frontiersOverTime.items():
            # print('frontiers: ', frontiers)
            frontiersOverTimeJson[task.name] = {i:str(frontier.entries[0].program) + '\n' + str(frontier.entries[0].logPosterior) for i,frontier in enumerate(frontiers) if len(frontier.entries) > 0}
        return frontiersOverTimeJson


    frontiersOverTime = convertFrontiersOverTimeToJson(frontierOverTime)
    # with open(resumePath + resumeDirectory + 'frontiersOverTime.json', 'w') as fp:
    #     json.dump(frontiersOverTime, fp)

    with open(resumePath + resumeDirectory + 'ecResults.json', 'w') as fp:
        json.dump({'learnedProductions':learnedProductions, 'frontiersOverTime':frontiersOverTime}, fp)

    # print(topDownGrammar)
    # print(firstFrontier)

    # # Tasks I'm not solving
    # # evaluateGrammars(firstFrontier, manuallySolvedTasks, grammar1=preConsolidationGrammar.insideOutside(firstFrontier, 30, iterations=1))
    # # evaluateGrammars(firstFrontier, manuallySolvedTasks, grammar1=preConsolidationGrammar.insideOutside(firstFrontier, 1))

    # # How does contextual model do?
    # # evaluateGrammars(firstFrontier, manuallySolvedTasks, grammar1=topDownGrammar, recognizer2=resumeRecognizer)

    # parent2UpdateProduction = {
    #     # (Primitive('blocks_to_original_grid', arrow(tblocks, tbool, tbool, tgridout),  None), 0): 'map_tbs',
    #     # (Primitive("map_tbs", arrow(ttbs, arrow(tblock, tblock, tblock), tbool, tblocks), None),1): 'move_until_touches_block',

    #     (Primitive('blocks_to_original_grid', arrow(tblocks, tbool, tbool, tgridout),  None), 0): 'map_blocks',
    #     # (Primitive("map_blocks", arrow(tblocks, arrow(tblock, tblock), tblocks), None),1): 'fill_color',
    #     (Primitive('fill_color', arrow(tblock, tcolor, tblock), None), 1): 'blue',
    #     # (Primitive("filter_blocks", arrow(tblocks, arrow(tblock, tbool), tblocks), None),1): 'touches_any_boundary',

    #     # (Primitive('blocks_to_original_grid', arrow(tblocks, tbool, tbool, tgridout),  None), 0): 'map_blocks',
    #     # (Primitive("map_tiles", arrow(ttiles, arrow(ttile, tblock), tblocks), None),1): 'extend_towards_until',
    #     # (Primitive("make_colorpair", arrow(tcolor, tcolor, tcolorpair), None), 1): 'invisible',

    #     # (Primitive('fill_color', arrow(tblock, tcolor, tblock), None), 1): 'teal',
    #     # (Primitive("map_blocks", arrow(tblocks, arrow(tblock, tblock), tblocks), None),1): 'fill_snakewise',
    #     # (Primitive("filter_template_block", arrow(tblocks, arrow(tblock, tbool), ttbs), None),0):'find_blocks_by_inferred_b',
    #     # (Primitive("filter_template_block", arrow(tblocks, arrow(tblock, tbool), ttbs), None),1):'has_min_tiles',
    #     # (Primitive("filter", arrow(tblocks, arrow(tblock, tblock), tblocks), None),0): 'extend_towards_until'
    # }
    # preConsolidationGrammarInsideOut = preConsolidationGrammar.insideOutside(firstFrontier,1)
    # contextualGrammar = ContextualGrammar.fromGrammar(preConsolidationGrammarInsideOut)
    # newContextualGrammar = upweightConditionalProductions(parent2UpdateProduction, 100, contextualGrammar)
    # evaluateGrammars(firstFrontier, manuallySolvedTasks, grammar1=preConsolidationGrammarInsideOut, grammar2=newContextualGrammar)

    # # recognitionModel = RecognitionModel(, baseGrammar)
    # # expandedFrontier = expandedFrontier()
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

    # # # def getTaskProgram(taskName):

    # # #     key = taskName.split('_')[1]
    # # #     return taskToProgram[key]

    # # # taskToProgram = {frontier.task.name:getTaskProgram(frontier) for frontier in expandedFrontier}

    # # # nnTrainTasks = [frontier.task for frontier in expandedFrontier]
    # # # for task,program in manuallySolvedTasks.items():
    # # #     if task not in taskToProgram:
    # # #         taskToProgram[task] = Program.parse(manuallySolvedTasks[task])


    # # trainedRecognizerPath = 'recognitionModels/2020-04-26 15:05:28.972185_trainTasks=2343_timeout=1200'
    # # with open(trainedRecognizerPath, 'rb') as handle:
    # #     trainedRecognizer = dill.load(handle)
    # # # print('{} Train Tasks'.format(len(nnTrainTasks)))
    # # # scoreTasks(trainedRecognizer, nnTrainTasks, taskToProgram, True)
    # # # print('\n {} Test Tasks'.format(len(nnTestTasks)))
    # # # scoreTasks(trainedRecognizer, nnTestTasks, taskToProgram, True)

    # explorationCompression(baseGrammar, trainTasks, testingTasks=testTasks, **args)
