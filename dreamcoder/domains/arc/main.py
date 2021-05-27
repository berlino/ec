from collections import defaultdict
import datetime
import dill
import json
import math
import numpy as np
import os
import sys
import pickle
import random
import signal
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamcoder.dreaming import helmholtzEnumeration
from dreamcoder.dreamcoder import explorationCompression, sleep_recognition, ecIterator
from dreamcoder.utilities import eprint, flatten, testTrainSplit, lse, runWithTimeout, pop_all_domain_specific_args
from dreamcoder.grammar import Grammar, ContextualGrammar
from dreamcoder.task import Task
from dreamcoder.type import Context, arrow, tbool, tlist, tint, t0, UnificationFailure
from dreamcoder.recognition import RecognitionModel, DummyFeatureExtractor, variable
from dreamcoder.program import Program
from dreamcoder.domains.arc.utilsPostProcessing import *
from dreamcoder.domains.arc.arcPrimitives import *
from dreamcoder.domains.arc.taskGeneration import *

import dreamcoder.domains.arc.language_utilities as language_utilities
from dreamcoder.domains.arc.language_model_feature_extractor import LMFeatureExtractor
from dreamcoder.domains.arc.cnn_feature_extractor import ArcCNN


DATA_DIR = "data/arc"

NONE = "NONE"
LANGUAGE_PROGRAMS_FILE = os.path.join(DATA_DIR, "best_programs_nl_sentences.csv") # Sentences and the best supervised programs.
TAGGED_LANGUAGE_FEATURES_FILE = os.path.join(DATA_DIR, "tagged_nl_sentences.csv") # Tagged semantic features.
LANGUAGE_ANNOTATIONS_FILE = os.path.join(DATA_DIR, "language/sentences/language.json") # All language annotations for training.
PRIMITIVE_HUMAN_READABLE = os.path.join(DATA_DIR, "primitiveNamesToDescriptions.json")
PRIOR_ENUMERATION_FRONTIERS = os.path.join(DATA_DIR, "prior_enumeration_frontiers_8hr.pkl")
ELICIT_FEATURE_VECTOR = os.path.join(DATA_DIR, "elicit_feature_vectors.json")

class LMPseudoTranslationFeatureExtractor(LMFeatureExtractor):
    """Generates pseudo annotations during training."""
    def __init__(self, tasks=[], testingTasks=[], cuda=False):
        return super(LMPseudoTranslationFeatureExtractor, self).__init__(tasks=tasks, testingTasks=testingTasks, cuda=cuda,use_language_model=False,primitive_names_to_descriptions=PRIMITIVE_HUMAN_READABLE,pseudo_translation_probability=0.5)

class LMCNNFeatureExtractor(LMFeatureExtractor):
    """Concatenates LM and CNN feature embeddings."""
    def __init__(self, tasks=[], testingTasks=[], cuda=False):
        return super(LMCNNFeatureExtractor, self).__init__(tasks=tasks, testingTasks=testingTasks, cuda=cuda,use_language_model=True,
        use_cnn=True)

class LMAugmentedFeatureExtractor(LMFeatureExtractor):
    """Concatenates LM and additional feature vector embeddings."""
    def __init__(self, tasks=[], testingTasks=[], cuda=False):
        return super(LMAugmentedFeatureExtractor, self).__init__(tasks=tasks, testingTasks=testingTasks, cuda=cuda,use_language_model=True,
        additional_feature_file=ELICIT_FEATURE_VECTOR,
        use_cnn=False)

class LMCNNPseudoFeatureExtractor(LMFeatureExtractor):
    """Concatenates LM and CNN feature embeddings with pseudo annotation tasks."""
    def __init__(self, tasks=[], testingTasks=[], cuda=False):
        return super(LMCNNPseudoFeatureExtractor, self).__init__(tasks=tasks, testingTasks=testingTasks, cuda=cuda,use_language_model=False,primitive_names_to_descriptions=PRIMITIVE_HUMAN_READABLE,pseudo_translation_probability=0.5,
        use_cnn=True)

class EvaluationTimeout(Exception):
    pass

class ArcTask(Task):
    def __init__(self, name, request, examples, evalExamples, features=None, cache=False, sentences=[]):
        super().__init__(name, request, examples, features=features, cache=cache)
        self.evalExamples = evalExamples
        self.sentences = sentences

    def checkEvalExamples(self, e, timeout=None):
        if timeout is not None:
            def timeoutCallBack(_1, _2): raise EvaluationTimeout()
        try:
            signal.signal(signal.SIGVTALRM, timeoutCallBack)
            signal.setitimer(signal.ITIMER_VIRTUAL, timeout)

            try:
                f = e.evaluate([])
            except IndexError:
                # free variable
                return False
            except Exception as e:
                eprint("Exception during evaluation:", e)
                return False

            for x, y in self.evalExamples:
                if self.cache and (x, e) in EVALUATIONTABLE:
                    p = EVALUATIONTABLE[(x, e)]
                else:
                    try:
                        p = self.predict(f, x)
                    except BaseException:
                        p = None
                    if self.cache:
                        EVALUATIONTABLE[(x, e)] = p
                if p != y:
                    if timeout is not None:
                        signal.signal(signal.SIGVTALRM, lambda *_: None)
                        signal.setitimer(signal.ITIMER_VIRTUAL, 0)
                    return False

            return True
        # except e:
            # eprint(e)
            # assert(False)
        except EvaluationTimeout:
            eprint("Timed out while evaluating", e)
            return False
        finally:
            if timeout is not None:
                signal.signal(signal.SIGVTALRM, lambda *_: None)
                signal.setitimer(signal.ITIMER_VIRTUAL, 0)

def retrieveARCJSONTasks(directory, useEvalExamplesForTraining=False, filenames=None):

    # directory = '/Users/theo/Development/program_induction/ec/ARC/data/training'
    data = []

    for filename in os.listdir(directory):
        if ("json" in filename):
            task = retrieveARCJSONTask(filename, directory, useEvalExamplesForTraining)
            if filenames is not None:
                if filename in filenames:
                    data.append(task)
            else:
                data.append(task)
    return data

def retrieveARCJSONTask(filename, directory, useEvalExamplesForTraining=False):
    with open(directory + "/" + filename, "r") as f:
        loaded = json.load(f)

    trainExamples = [
            ((Grid(gridArray=example["input"]),), Grid(gridArray=example["output"]))
            for example in loaded["train"]
        ]
    evalExamples = [
            ((Grid(gridArray=example["input"]),), Grid(gridArray=example["output"]))
            for example in loaded["test"]
        ]

    if useEvalExamplesForTraining:
        trainExamples = trainExamples + evalExamples
        evalExamples = []

    task = ArcTask(
        filename,
        arrow(tgridin, tgridout),
        trainExamples,
        evalExamples
    )
    task.specialTask = ('arc', 5)
    return task

def preload_initial_frontiers(preload_frontiers_file):
    with open(preload_frontiers_file, "rb") as f:
        preloaded_frontiers = pickle.load(f)
    tasks_to_preloaded_frontiers = {
        task.name : frontier
        for task, frontier in preloaded_frontiers.items() if not frontier.empty
    }
    print(f"Preloaded frontiers for {len(tasks_to_preloaded_frontiers)} tasks.")
    return tasks_to_preloaded_frontiers

def arc_options(parser):
    # parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--singleTask", default=False, action="store_true")
    parser.add_argument("--unigramEnumerationTimeout", type=int, default=3600)
    parser.add_argument("--firstTimeEnumerationTimeout", type=int, default=1)
    parser.add_argument("--featureExtractor", default="dummy", choices=[
        "arcCNN",
        "dummy",
        "LMFeatureExtractor",
        "LMPseudoTranslationFeatureExtractor",
        "LMCNNFeatureExtractor",
        "LMCNNPseudoFeatureExtractor",
        "LMAugmentedFeatureExtractor"])

    parser.add_argument("--primitives", default="rich", help="Which primitive set to use", choices=["base", "rich"])
    parser.add_argument("--test_language_models", action="store_true")
    parser.add_argument("--test_language_dc_recognition", action="store_true")
    parser.add_argument("--language_encoder", help="Which language encoder to use for test_language_models.")
    parser.add_argument("--language_program_data", default=LANGUAGE_PROGRAMS_FILE)
    parser.add_argument("--language_annotations_data", default=LANGUAGE_ANNOTATIONS_FILE)
    parser.add_argument("--tagged_annotations_data", default=TAGGED_LANGUAGE_FEATURES_FILE)
    parser.add_argument("--primitive_names_to_descriptions",
    default=PRIMITIVE_HUMAN_READABLE)
    parser.add_argument("--preload_frontiers", default=NONE)

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

def gridToArray(grid):
    temp = np.full((grid.getNumRows(),grid.getNumCols()),None)
    for yPos,xPos in grid.points:
        temp[yPos, xPos] = str(grid.points[(yPos,xPos)])
    return temp

def run_tests(args):
    if args.pop("test_language_models"):
        from dreamcoder.domains.arc.test_language_models import main
        main(args)
        sys.exit(0)
    if args.pop("test_language_dc_recognition"):
        from dreamcoder.domains.arc.test_language_dc_recognition import main
        main(args)
        sys.exit(0)

def main(args):
    """
    Takes the return value of the `commandlineArguments()` function as input and
    trains/tests the model on manipulating sequences of numbers.

    """
    # samples = {
    #     "007bbfb7.json": _solve007bbfb7,
    #     "c9e6f938.json": _solvec9e6f938,
    #     "50cb2852.json": lambda grid: _solve50cb2852(grid)(8),
    #     "fcb5c309.json": _solvefcb5c309,
    #     "97999447.json": _solve97999447,
    #     "f25fbde4.json": _solvef25fbde4,
    #     "72ca375d.json": _solve72ca375d,
    #     "5521c0d9.json": _solve5521c0d9,
    #     "ce4f8723.json": _solvece4f8723,
    # }
    # 
    run_tests(args)

    import os

    homeDirectory = "/".join(os.path.abspath(__file__).split("/")[:-4])
    dataDirectory = homeDirectory + "/arc_data/data/"

    trainTasks = retrieveARCJSONTasks(dataDirectory + 'training', useEvalExamplesForTraining=True, filenames=None)
    holdoutTasks = retrieveARCJSONTasks(dataDirectory + 'evaluation')
    
    language_annotations_data = args.pop("language_annotations_data") 
    if language_annotations_data is not None:
        trainTasks, holdoutTasks = language_utilities.add_task_language_annotations(trainTasks, holdoutTasks, language_annotations_data)
        
    # Load any pre-initialized frontiers.
    preloaded_frontiers_file = args.pop("preload_frontiers")
    preloaded_frontiers = dict()
    if preloaded_frontiers_file != NONE:
        preloaded_frontiers = preload_initial_frontiers(preloaded_frontiers_file)

    primitivesTable = {
        "base": basePrimitives() + leafPrimitives(),
        "rich": basePrimitives() + leafPrimitives() + moreSpecificPrimitives()
        }

    baseGrammar = Grammar.uniform(primitivesTable[args.pop("primitives")])
    # print("base Grammar {}".format(baseGrammar))

    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "experimentOutputs/arc/%s" % timestamp
    os.system("mkdir -p %s" % outputDirectory)

    args.update(
        {"outputPrefix": "%s/arc" % outputDirectory, "evaluationTimeout": 1,}
    )

    featureExtractor = {
        "dummy": DummyFeatureExtractor,
        "arcCNN": ArcCNN,
        "LMFeatureExtractor" : LMFeatureExtractor,
        "LMPseudoTranslationFeatureExtractor" : LMPseudoTranslationFeatureExtractor,
        "LMCNNFeatureExtractor" : LMCNNFeatureExtractor,
        "LMCNNPseudoFeatureExtractor" : LMCNNPseudoFeatureExtractor,
        "LMAugmentedFeatureExtractor" : LMAugmentedFeatureExtractor
    }[args.pop("featureExtractor")]

    if args.pop("singleTask"):
        trainTasks = [trainTasks[0]]

    # Utility function to remove any command line arguments that are not in the main iterator.
    pop_all_domain_specific_args(args, ecIterator)
    explorationCompression(baseGrammar, trainTasks, featureExtractor=featureExtractor, testingTasks=[], 
    preloaded_frontiers=preloaded_frontiers,
     **args)
