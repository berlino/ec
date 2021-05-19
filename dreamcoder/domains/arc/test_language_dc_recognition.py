"""
test_language_dc_recognition.py | Tests for DreamCoder-style recognition models initialized with language data.

Requires tests to be manually added to a main.

Usage:
    python bin/arc.py
        --test_language_dc_recognition
"""
import os
import csv
import json
from collections import defaultdict

from dreamcoder.type import arrow
from dreamcoder.domains.arc.arcPrimitives import *
from dreamcoder.task import Task
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.recognition import RecognitionModel, DummyFeatureExtractor

from dreamcoder.domains.arc.language_model_feature_extractor import LMFeatureExtractor

ARC_REQUEST = arrow(tgridin, tgridout)

def build_language_tasks(language_program_data):
    """:ret: Initialized grammar; {ArcTask : Frontier} annotated with language"""
    # Initialize base primitives to parse the programs.
    basePrimitives()
    leafPrimitives()
    arc_grammar = Grammar.uniform(basePrimitives() + leafPrimitives())
    
    task_names_to_tasks = defaultdict(list)
    tasks_to_frontiers = {}
    with open(language_program_data, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name, program, sentences = load_sentence_data(row)
            if task_name is not None:
                program = Program.parse(program)
                # Create separate tasks for each language annotations
                task_duplicate_id = len(task_names_to_tasks[task_name])
                task_full_name = f"{task_name}_{task_duplicate_id}"
                task = Task(name=task_full_name, request=ARC_REQUEST, examples=[])
                task.sentences = sentences
                # Create a separate frontier per task for now.
                tasks_to_frontiers[task] = Frontier.makeEmpty(task)
                tasks_to_frontiers[task].entries.append(
                    FrontierEntry(program=program,
                                  logLikelihood=0.0,
                                  logPrior=0.0))
                task_names_to_tasks[task_name].append(task)
    print(f"Loaded: {len(task_names_to_tasks)} unique tasks | {len(tasks_to_frontiers)} annotated tasks.")
    return arc_grammar, task_names_to_tasks, tasks_to_frontiers

def load_sentence_data(row):
    """Loads sentence-delimited data.
    Returns task_name, program, [array of sentences]
    """
    if row['phrase_kind'] != 'output':
        return None, None, None
    else:
        task_name = row[""]
        program = row["program"]
        sentences = row['natural_language'].split("||")
        return task_name, program, sentences

def leave_one_out_evaluation_dc_model(arc_grammar, task_names_to_tasks, tasks_to_frontiers, recognition_model_fn, train_recognition_model_fn, evaluate_likelihood_fn):
    print(f"Running leave one out evaluation on {len(task_names_to_tasks)} tasks.")
    tasks_to_likelihoods = dict()
    for idx, task_name in enumerate(task_names_to_tasks):
        tasks_for_heldout_task_name = task_names_to_tasks[task_name]
        training_tasks = {t : d for t, d in tasks_to_frontiers.items() if t not in tasks_for_heldout_task_name}
        recognition_model = recognition_model_fn()
        train_recognition_model_fn(recognition_model, training_tasks)
        task_likelihoods = []
        for heldout_task in tasks_for_heldout_task_name:
            heldout_program = tasks_to_frontiers[heldout_task].bestPosterior.program
            log_likelihood = evaluate_likelihood_fn(recognition_model, heldout_task, heldout_program)
            task_likelihoods.append(log_likelihood)
        tasks_to_likelihoods[task_name] = np.mean(task_likelihoods)
        if idx % 10 == 0:
            print(f"Evaluated: {idx}/{len(task_names_to_tasks)}")
            print(f"Current average log likelihood: {np.mean(list(tasks_to_likelihoods.values()))}")
    print(f"Average log likelihood: {np.mean(list(tasks_to_likelihoods.values()))}")
    return tasks_to_likelihoods

def test_leave_one_out_bigram_dc_dummy_model(arc_grammar, task_names_to_tasks, tasks_to_frontiers):
    return test_leave_one_out_dc_lm_model(arc_grammar, task_names_to_tasks, tasks_to_frontiers, contextual=True, dummy=True)

def test_leave_one_out_bigram_dc_lm_model(arc_grammar, task_names_to_tasks, tasks_to_frontiers):
    return test_leave_one_out_dc_lm_model(arc_grammar, task_names_to_tasks, tasks_to_frontiers, contextual=True)

def test_leave_one_out_unigram_dc_lm_model(arc_grammar, task_names_to_tasks, tasks_to_frontiers):
    return test_leave_one_out_dc_lm_model(arc_grammar, task_names_to_tasks, tasks_to_frontiers, contextual=False)

def test_leave_one_out_dc_lm_model(arc_grammar, task_names_to_tasks, tasks_to_frontiers, contextual, dummy=False):
    def recognition_model_fn():
        if dummy:
            feature_extractor = DummyFeatureExtractor(tasks=None)
        else:
            feature_extractor = LMFeatureExtractor()
        recognition_model = RecognitionModel(
            featureExtractor=feature_extractor,
            grammar=arc_grammar,
            contextual=contextual)
        return recognition_model
        
    def train_recognition_model_fn(recognition_model, training_tasks):
        recognition_model.train(
            frontiers=training_tasks.values(),
            steps=1000,
            helmholtzRatio=0,
            biasOptimal=True,
            vectorized=True
        )
    
    def evaluate_likelihood_fn(recognition_model, heldout_task, heldout_program):
        predicted_grammar = recognition_model.grammarOfTask(heldout_task).untorch()
        ll = predicted_grammar.logLikelihood(request=ARC_REQUEST, expression=heldout_program)
        return ll
    
    leave_one_out_evaluation_dc_model(arc_grammar, task_names_to_tasks, tasks_to_frontiers, recognition_model_fn, train_recognition_model_fn, evaluate_likelihood_fn)

def main(args):
    print(f"Running test_language_dc_recognition.....")
    arc_grammar, task_names_to_tasks, tasks_to_frontiers = build_language_tasks(language_program_data=args["language_program_data"])
    
    # test_leave_one_out_unigram_dc_lm_model(arc_grammar, task_names_to_tasks, tasks_to_frontiers)
    # test_leave_one_out_bigram_dc_lm_model(arc_grammar, task_names_to_tasks, tasks_to_frontiers)
    test_leave_one_out_bigram_dc_dummy_model(arc_grammar, task_names_to_tasks, tasks_to_frontiers)
    