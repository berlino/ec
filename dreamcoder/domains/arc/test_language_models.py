"""
test_language_models.py 
Exploratory work setting up language models for the ARC domain.

Usage:
    python bin/arc.py 
        --test_language_models
        --language_encoder t5_linear_predictor # Model name: looks up a model in the model registry.
"""
import os
import numpy as np
import csv
from collections import defaultdict, Counter
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset

import tensorflow as tf
from transformers import pipeline, T5Tokenizer, TFT5EncoderModel

from dreamcoder.program import Program
from dreamcoder.grammar import Grammar, ContextualGrammar
from dreamcoder.type import arrow
from dreamcoder.domains.arc.arcPrimitives import *
from dreamcoder.task import Task
from dreamcoder.frontier import Frontier, FrontierEntry

DATA_DIR = "data/arc"
LANGUAGE_DATA_FILE = "ManyProgramsPlusNlDescription.csv"

# Prediction models supported in this file.
UNIFORM_UNIGRAM_PRIOR = "uniform_unigram_prior" # Uniform distribution over unigrams.
FITTED_UNIGRAM_PRIOR = "fitted_unigram_prior" # Frequency distribution over unigrams.
T5_LINEAR_MODEL = "t5_linear_predictor" 

T5_MODEL = 't5-small' 
ROBERTA_MODEL = 'distilroberta-base'
T5 = 't5'

ARC_REQUEST = arrow(tgridin, tgridout)

def load_language_program_data(language_file):
    """
    Loads language-program data. Parses programs into Program objects under the ARC DSL.
    Returns: 
        arc_grammar: uniform grammar containing the DSL primitives.
        language_program_data: {task_id_{COUNTER}: (language, Program)}.
    """
    # Initialize base primitives to parse the programs.
    basePrimitives()
    leafPrimitives()
    arc_grammar = Grammar.uniform(basePrimitives() + leafPrimitives())
    
    # Load the language data and the parsed programs.
    full_filepath = os.path.join(DATA_DIR, language_file)
    task_counter = Counter()
    language_program_data = dict()
    frontiers = {}
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

def leave_one_out_evaluation(task_to_data, model_class, grammar):
    print(f"Running leave one out evaluation on {len(task_to_data)} tasks.")
    tasks_to_likelihoods = dict()
    for idx, task in enumerate(task_to_data):
        program, language  = task_to_data[task]
        training_tasks = {t : d for t, d in task_to_data.items() if t != task}
        model = model_class()
        model.fit(training_tasks, grammar)
        likelihood = model.evaluate_likelihood(language, program)
        tasks_to_likelihoods[task] = likelihood
        if idx % 10 == 0:
            print(f"Evaluated: {idx}/{len(task_to_data)}")
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
    return MODEL_REGISTRY[model_tag]

class UnigramDataset(Dataset):
    def __init__(self, inputs, outputs=None):
        self.X = inputs
        self.y = outputs
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]
        
class LinearUnigramModel(nn.Module):
    """Linear classifier from language model sentence embeddings to unigrams in the base grammar."""
    def __init__(self, lm_model_name):
        self.lm_model_name = lm_model_name
        if T5 in self.lm_model_name:
            self.tokenizer = T5Tokenizer.from_pretrained(self.lm_model_name)
            self.t5_encoder_model = TFT5EncoderModel.from_pretrained(self.lm_model_name)
            self.featurizer = self._t5_featurizer_fn
        else:
            self.featurizer = pipeline('feature-extraction', self.lm_model_name)
        
        self.base_grammar = None
        self.primitive_names = None
        self.primitive_name_to_idx = None
        self.idx_to_primitive_name = None 
        self.linear = None
        self.input_dim = None
        self.output_dim = None
        super(LinearUnigramModel, self).__init__()
            
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
    
    def _featurize_language(self, language):
        """Featurizes the language. 
        language: [n_language array of sentences.]
        :ret: torch array of size n_language x <HIDDEN_STATE_DIM>
        """
        outputs = self.featurizer(language)
        return torch.from_numpy(outputs).type(torch.FloatTensor)
    
    def _program_to_unigram_probabilities(self, program, grammar):
        """Program to normalized unigram probabilities over the grammar. Simplified unigram probabilities, based on unigram counts excluding variables."""
        unigram_probabilities = np.zeros(len(self.primitive_names))
        unigrams = program.left_order_tokens()
        for u in unigrams:
            unigram_probabilities[self.primitive_name_to_idx[u]] += 1
        unigram_probabilities /= np.sum(unigram_probabilities)
        return unigram_probabilities
        
    def _programs_to_unigram_probabilities(self, programs, grammar):
        """
        Featurizes programs into a unigram probability summary over the DSL of primitives
        programs: [n_programs array of programs]
        grammar: DreamCoder Grammar containing all DSL primitives.
        :ret: np.array of size n_programs x <number of primitives>.
        """
        unigram_probabilities = []
        for program in programs:
            unigram_probability = self._program_to_unigram_probabilities(program, grammar)
            unigram_probabilities.append(unigram_probability)
        unigram_probabilities = np.stack(unigram_probabilities)
        unigram_probabilities = torch.from_numpy(unigram_probabilities).type(torch.FloatTensor)
        return unigram_probabilities
    
    def _initialize_base_unigram_grammar(self, grammar):
        self.base_grammar = grammar
        self.primitive_names = [str(production[-1]) for production in self.base_grammar.productions]
        self.primitive_name_to_idx, self.idx_to_primitive_name = {}, {}
        for (idx, name) in enumerate(self.primitive_names):
            self.primitive_name_to_idx[name] = idx
            self.idx_to_primitive_name[idx] = name
        self.output_dim = len(self.primitive_names)
    
    def _train(self, inputs, outputs, epochs=10):
        batch_size = 50
        lr_rate = 0.001
        self.loss = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr_rate)
        
        train_dataset = UnigramDataset(inputs, outputs)
        
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs): 
            for i, (inputs, outputs) in enumerate(train_loader):
                inputs, outputs = Variable(inputs), Variable(outputs)
                self.optimizer.zero_grad()
                predictions = self.linear(inputs)
                loss = self.loss(predictions, outputs)
                loss.backward()
                self.optimizer.step()
    
    def fit(self, task_to_data, grammar):
        """
        task_to_data: dict from task_names to (program, language) for each task.
        """
        self._initialize_base_unigram_grammar(grammar)
        programs, language = zip(*task_to_data.values())
        unigram_probabilities = self._programs_to_unigram_probabilities(programs, grammar)
        featurized_language = self._featurize_language(language)
        self.input_dim = featurized_language.shape[-1]
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        self._train(featurized_language, unigram_probabilities)
    
    def evaluate_likelihood(self, language, ground_truth_program):
        """Evaluate likelihood of a ground truth program under predicted unigram probabilities."""
        featurized_language = self._featurize_language([language])
        predicted_probabilities = self.linear(featurized_language).detach()[0]
        unigram_grammar = Grammar(0.0, #logVariable
                       [(float(predicted_probabilities[k]), t, unigram)
                        for k, (_, t, unigram) in enumerate(self.base_grammar.productions)],
                       continuationType=self.base_grammar.continuationType)
        ll = unigram_grammar.logLikelihood(request=ARC_REQUEST, expression=ground_truth_program)
        return ll

@register_model(T5_LINEAR_MODEL)
class T5LinearUnigramModel(LinearUnigramModel):
    def __init__(self):
        super(T5LinearUnigramModel, self).__init__(lm_model_name=T5_MODEL) 

class BaselinePriorUnigramModel():
    """Baseline prior models. Fits a single unigram grammar that is not contextually re-calculated based each task."""
    def __init__(self):
        self.prior_grammar = None
        
    def fit(self, task_to_data, grammar):
        print("Unimplemented in the base class.")
        assert False
    
    def evaluate_likelihood(self, language, ground_truth_program):
        return self.prior_grammar.logLikelihood(request=ARC_REQUEST, expression=ground_truth_program)

@register_model(UNIFORM_UNIGRAM_PRIOR)
class UniformPriorUnigramModel(BaselinePriorUnigramModel):
    """Uniform prior over the unigram grammar."""
    def __init__(self):
        super(UniformPriorUnigramModel, self).__init__()
    
    def fit(self, task_to_data, grammar):
        self.prior_grammar = grammar

@register_model(FITTED_UNIGRAM_PRIOR)
class FittedUnigramPriorModel(BaselinePriorUnigramModel):
    def __init__(self):
        super(FittedUnigramPriorModel, self).__init__()
    
    def fit(self, task_to_data, grammar):
        # Fit grammar using the inside outside algorithm to all of the frontiers.
        task_names_to_tasks = {}
        tasks_to_frontiers = {}
        for task_name, (program, _) in task_to_data.items():
            original_task_name = task_name.split("_")[0]
            if original_task_name not in task_names_to_tasks:
                task = Task(name=original_task_name, request=ARC_REQUEST, examples=[])
                task_names_to_tasks[original_task_name] = task
                tasks_to_frontiers[task] = Frontier.makeEmpty(task)
            task = task_names_to_tasks[original_task_name]
            tasks_to_frontiers[task].entries.append(
                FrontierEntry(program=program,
                              logLikelihood=0.0,
                              logPrior=0.0))
        # Grammar inside outside.
        self.prior_grammar = grammar.insideOutside(frontiers=tasks_to_frontiers.values(), pseudoCounts=1)            
    
def main(args):
    print("Running language model evaluations....")
    arc_grammar, language_program_data = load_language_program_data(language_file=LANGUAGE_DATA_FILE)
    model = get_model(model_tag=args["language_encoder"])
    leave_one_out_evaluation(language_program_data, model, arc_grammar)
    # TBD allow it to run leave one out on the tasks vs. the sentences.