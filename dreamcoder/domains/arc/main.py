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
from dreamcoder.domains.arc.arcPrimitives import *

import dreamcoder.domains.arc.language_utilities as language_utilities
from dreamcoder.domains.arc.language_model_feature_extractor import LMFeatureExtractor
from dreamcoder.domains.arc.cnn_feature_extractor import ArcCNN


DATA_DIR = "data/arc"
LARC_DIR = "data/larc/"

CHECKPOINT_FILE_PREFIX = "experimentOutputs"
NONE = "NONE"
LANGUAGE_PROGRAMS_FILE = os.path.join(DATA_DIR, "best_programs_nl_sentences.csv") # Sentences and the best supervised programs.
TAGGED_LANGUAGE_FEATURES_FILE = os.path.join(DATA_DIR, "tagged_nl_sentences.csv") # Tagged semantic features.
LANGUAGE_ANNOTATIONS_FILE = os.path.join(DATA_DIR, "language/sentences/language.json") # All language annotations for training.
PRIMITIVE_HUMAN_READABLE = os.path.join(DATA_DIR, "primitiveNamesToDescriptions.json")
PRIOR_ENUMERATION_FRONTIERS = os.path.join(DATA_DIR, "prior_enumeration_frontiers_8hr.pkl")
ELICIT_FEATURE_VECTOR = os.path.join(DATA_DIR, "elicit_feature_vectors.json")
TRAIN_TEST_SPLIT_FILENAME = "train_test_split.json"

SPLIT_SEED = 0
SPLIT_RATIO = 0.5

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
    # passed in json message to ocaml solver; used to select the arc specific solver
    task.specialTask = ('arc', None)
    return task

def train_test_split(task_names, ratio, seed):
    
    random.seed(seed)
    random.shuffle(task_names)
    train_size = int(ratio * len(task_names))
    # change global seed so that it's not always fixed for other parts of the pipeline
    random.seed()

    return {"train": task_names[:train_size], "test":task_names[train_size:]}

def preload_initial_frontiers(preload_frontiers_file, is_checkpoint_file=False):

    if is_checkpoint_file:
        with open(preload_frontiers_file, "rb") as handle:
            result = dill.load(handle)
            preloaded_frontiers = result.allFrontiers


    else:
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
    parser.add_argument("--splitRatio", type=float, default=0.5)
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
    parser.add_argument("--filter_test_task_if_no_nl", default=False, action="store_true")
    # parser.add_argument("-i", type=int, default=10)

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
    run_tests(args)

    import os

    homeDirectory = "/".join(os.path.abspath(__file__).split("/")[:-4])
    dataDirectory = homeDirectory + "/arc_data/data/"

    # load tasks
    tasks_with_eval_ex = retrieveARCJSONTasks(dataDirectory + 'training', useEvalExamplesForTraining=True, filenames=None)
    tasks_without_eval_ex = retrieveARCJSONTasks(dataDirectory + 'training', useEvalExamplesForTraining=False, filenames=None)

    # load train and test task names
    train_test_split_dict = json.load(open(LARC_DIR + TRAIN_TEST_SPLIT_FILENAME, "r"))
    train_task_names = [t for t in train_test_split_dict["train"]]
    test_task_names = [t for t in train_test_split_dict["test"]]
 
    # when evaluating we induce program only from training examples
    trainTasks = [t for t in tasks_with_eval_ex if t.name in train_task_names]
    testTasks = [t for t in tasks_without_eval_ex if t.name in test_task_names]
    
    language_annotations_data = args.pop("language_annotations_data") 
    if language_annotations_data is not None:
        trainTasks, testTasks = language_utilities.add_task_language_annotations(trainTasks, testTasks, language_annotations_data)

    filter_test_task_if_no_nl = args.pop("filter_test_task_if_no_nl")
    if filter_test_task_if_no_nl:
        testTasks = [t for t in testTasks if len(t.sentences) > 0]
    print("{} test tasks".format(len(testTasks)))
 
    # Load any pre-initialized frontiers.
    preloaded_frontiers_file = args.pop("preload_frontiers")
    preloaded_frontiers = dict()
    if preloaded_frontiers_file != NONE:
        is_checkpoint_file = CHECKPOINT_FILE_PREFIX in preloaded_frontiers_file
        preloaded_frontiers = preload_initial_frontiers(preloaded_frontiers_file, is_checkpoint_file)

    assert len([t for t,f in preloaded_frontiers.items() if (len(f.entries) > 0) and (t in train_task_names)]) == len(preloaded_frontiers)

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
    explorationCompression(baseGrammar, trainTasks, featureExtractor=featureExtractor, testingTasks=testTasks, 
    preloaded_frontiers=preloaded_frontiers,
     **args)
