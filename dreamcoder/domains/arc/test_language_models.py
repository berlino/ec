"""
test_language_models.py 
Exploratory work setting up language models for the ARC domain.

Usage:
    python bin/arc.py 
        --test_language_models
        --language_encoder t5_linear_predictor # Model name
"""
import os
import numpy as np
import csv
from collections import defaultdict, Counter

import tensorflow as tf
from transformers import pipeline, T5Tokenizer, TFT5EncoderModel

from dreamcoder.program import Program
from dreamcoder.grammar import Grammar, ContextualGrammar
from dreamcoder.type import arrow
from dreamcoder.domains.arc.arcPrimitives import *

DATA_DIR = "data/arc"
LANGUAGE_DATA_FILE = "ManyProgramsPlusNlDescription.csv"

# Prediction models supported in this file.
T5_LINEAR_MODEL = "t5_linear_predictor" 

T5_MODEL = 't5-small' 
ROBERTA_MODEL = 'distilroberta-base'
T5 = 't5'

def load_language_program_data(language_file):
    """
    Loads language-program data. Parses programs into Program objects under the ARC DSL.
    Returns: {task_id_{COUNTER}: (language, Program)}.
    """
    # Initialize base primitives to parse the programs.
    basePrimitives()
    leafPrimitives()
    arc_grammar = Grammar.uniform(basePrimitives() + leafPrimitives())
    
    # Load the language data and the parsed programs.
    full_filepath = os.path.join(DATA_DIR, language_file)
    task_counter = Counter()
    language_program_data = dict()
    with open(full_filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            original_task_name, program, language = row["taskName"], row['program'], row['nlDescription']
            counter = task_counter[original_task_name]
            task_name = f"{original_task_name}_{counter}"
            program = Program.parse(program)
            language_program_data[task_name] = (program, language)
            task_counter[original_task_name] += 1
    return arc_grammar, language_program_data 

def leave_one_out_evaluation(task_to_data, model, grammar):
    print(f"Running leave one out evaluation on {len(task_to_data)} tasks.")
    tasks_to_likelihoods = dict()
    for task in task_to_data:
        program, ground_truth_program  = task_to_data[task]
        training_tasks = {t : d for t, d in task_to_data.items() if t != task}
        model.fit(training_tasks, grammar)
        likelihood = model.evaluate_likelihood(language, ground_truth_program, grammar)
        tasks_to_likelihoods[task] = likelihood
    print(f"Average likelihood: {np.mean(list(tasks_to_likelihoods.values()))}")
    return tasks_to_likelihoods


## Models: defines models that can be passed directly to the evaluation. use @register_model to register models.
MODEL_REGISTRY = dict()
def register_model(name):
    def wrapper(c):
        MODEL_REGISTRY[name] = c
        return c
    return wrapper

def get_model(model_tag):
    return MODEL_REGISTRY[model_tag]()

class LinearUnigramModel():
    def __init__(self, lm_model_name):
        self.lm_model_name = lm_model_name
        if T5 in self.lm_model_name:
            self.tokenizer = T5Tokenizer.from_pretrained(self.lm_model_name)
            self.t5_encoder_model = TFT5EncoderModel.from_pretrained(self.lm_model_name)
            self.featurizer = self._t5_featurizer_fn
        else:
            self.featurizer = pipeline('feature-extraction', self.lm_model_name)
            
    def _t5_featurizer_fn(self, language):
        """Featurizes batch of sentences using a mean over the tokens in each sentence.
        args:
            language: [array of N sentences]
        ret: numpy array of size N x <HIDDEN_STATE_DIM>
        """
        input_ids = self.tokenizer(language, return_tensors="tf", padding=True, truncation=True).input_ids  # Batch size 1
        outputs = self.t5_encoder_model(input_ids)
        last_hidden_states = outputs.last_hidden_state 
        reduced = tf.math.reduce_mean(last_hidden_states, axis=1)
        return reduced.numpy()
    
    def _program_unigram_likelihoods(self, programs, grammar):
        """
        Featurizes the programs. Simplified unigram likelihoods, based on unigram counts excluding variables.
        Returns: a numpy array of size N x <number of primitives>/
        """
        for program in programs:
            tokens = program.left_order_tokens()
            import pdb; pdb.set_trace()
    
    def _featurize_language(self, language):
        """Featurizes the language. 
        Returns: numpy array of size N x <HIDDEN_STATE_DIM>
        """
        outputs = self.featurizer(language)
        return outputs
    
    def fit(self, task_to_data, grammar):
        """
        task_to_data: dict from task_names to (program, language) for each task.
        """
        programs, language = zip(*task_to_data.values())
        
        featurized_language = self._featurize_language(language)
        print(featurized_language.shape)
        
        # unigram_likelihoods = self. _program_unigram_likelihoods(programs, grammar)
        
        import pdb; pdb.set_trace()

@register_model(T5_LINEAR_MODEL)
class T5LinearUnigramModel(LinearUnigramModel):
    def __init__(self):
        super(T5LinearUnigramModel, self).__init__(lm_model_name=T5_MODEL)    
    
def main(args):
    print("Running language model evaluations....")
    arc_grammar, language_program_data = load_language_program_data(language_file=LANGUAGE_DATA_FILE)
    model = get_model(model_tag=args["language_encoder"])
    leave_one_out_evaluation(language_program_data, model, arc_grammar)
    