"""
test_language_models.py 
Exploratory work setting up language models for the ARC domain.

Usage:
    python bin/arc.py 
        --test_language_models
        --language_encoder t5_linear_predictor # Model name: looks up a model in the model registry.
    
    Baseline results:
        uniform_unigram_prior : -30.729733065225524
        fitted_unigram_prior : -19.092839837180854
        t5_linear_predictor, all sentences at once : -30.753040922853256
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
from transformers import pipeline, T5Tokenizer, T5EncoderModel

from dreamcoder.program import Program
from dreamcoder.grammar import Grammar, ContextualGrammar
from dreamcoder.type import arrow
from dreamcoder.domains.arc.arcPrimitives import *
from dreamcoder.task import Task
from dreamcoder.frontier import Frontier, FrontierEntry

DATA_DIR = "data/arc"
LANGUAGE_PROGRAMS_FILE = "ManyProgramsPlusNlDescription.csv"
PRIMITIVE_NATURAL_LANGUAGE_FILE = "primitiveNamesToDescriptions.json"

# Prediction models supported in this file.
UNIFORM_UNIGRAM_PRIOR = "uniform_unigram_prior" # Uniform distribution over unigrams.
FITTED_UNIGRAM_PRIOR = "fitted_unigram_prior" # Frequency distribution over unigrams.
T5_LINEAR_MODEL = "t5_linear_predictor" # Linear predictor from language-model embeddings to unigrams.
T5_SIMILARITY_MODEL = "t5_similarity_model" # Normalized similarities between language-model-embeddings and unigram names.

T5_MODEL = 't5-small' 
ROBERTA_MODEL = 'distilroberta-base'
T5 = 't5'

ARC_REQUEST = arrow(tgridin, tgridout)

def load_language_program_data(language_programs_file):
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
    full_filepath = os.path.join(DATA_DIR, language_programs_file)
    language_program_data = defaultdict(list)
    with open(full_filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name, program, language = row["taskName"], row['program'], row['nlDescription']
            program = Program.parse(program)
            language_program_data[task_name].append((program, language))
    return arc_grammar, language_program_data 

def leave_one_out_evaluation(task_to_data, model_class, grammar):
    """Leave one out evaluation on a per-task basis. Fits a model to all of the other tasks, and then computes the average likelihood for the remaining language/program data for the left out task."""
    print(f"Running leave one out evaluation on {len(task_to_data)} tasks.")
    tasks_to_likelihoods = dict()
    for idx, task in enumerate(task_to_data):
        heldout_task_programs_and_language  = task_to_data[task]
        training_tasks = {t : d for t, d in task_to_data.items() if t != task}
        model = model_class()
        model.fit(training_tasks, grammar)
        task_likelihoods = []
        for (program, language) in heldout_task_programs_and_language:
            log_likelihood = model.evaluate_likelihood(language, program)
            task_likelihoods.append(log_likelihood)
        tasks_to_likelihoods[task] = np.mean(task_likelihoods)
        if idx % 10 == 0:
            print(f"Evaluated: {idx}/{len(task_to_data)}")
    print(f"Average log likelihood: {np.mean(list(tasks_to_likelihoods.values()))}")
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
        for task_name, task_programs_and_language in task_to_data.items():
            original_task_name = task_name.split("_")[0]
            if original_task_name not in task_names_to_tasks:
                task = Task(name=original_task_name, request=ARC_REQUEST, examples=[])
                task_names_to_tasks[original_task_name] = task
                tasks_to_frontiers[task] = Frontier.makeEmpty(task)
            task = task_names_to_tasks[original_task_name]
            for (program, _) in task_programs_and_language:
                tasks_to_frontiers[task].entries.append(
                    FrontierEntry(program=program,
                                  logLikelihood=0.0,
                                  logPrior=0.0))
        # Grammar inside outside.
        self.prior_grammar = grammar.insideOutside(frontiers=tasks_to_frontiers.values(), pseudoCounts=1)            
    
class UnigramDataset(Dataset):
    def __init__(self, inputs, outputs=None):
        self.X = inputs
        self.y = outputs
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]

class LanguageModelUnigramModel(nn.Module):
    """Base class for models that embed the natural language descriptions using
    a pre-initialized language model, and output distributions over unigrams in 
    a program DSL."""
    def __init__(self, lm_model_name):
        super(LanguageModelUnigramModel, self).__init__()
        self.lm_model_name = lm_model_name
        if T5 in self.lm_model_name:
            self.tokenizer = T5Tokenizer.from_pretrained(self.lm_model_name)
            self.t5_encoder_model = T5EncoderModel.from_pretrained(self.lm_model_name)
            self.language_encoder_fn = self._t5_language_encoder_fn
        else:
            self.language_encoder_fn = pipeline('feature-extraction', self.lm_model_name)
        
        self.input_dim = None
        self.output_dim = None
    
    def _t5_language_encoder_fn(self, language):
        """Featurizes a batch of sentences using the T5 model. 
        Each sentence is embedded as a mean over the hidden states of its contextual token embeddings.
        args:
            language: [array of N sentences]
        ret: numpy array of size N x <HIDDEN_STATE_DIM>
        """
        input_ids = self.tokenizer(language, return_tensors="pt", padding=True, truncation=True).input_ids  
        outputs = self.t5_encoder_model(input_ids)
        last_hidden_states = outputs.last_hidden_state 
        reduced = torch.mean(last_hidden_states, dim=1).detach()
        return reduced
        
class LinearUnigramModel(LanguageModelUnigramModel):
    """Linear classifier from language model sentence embeddings to unigrams in the base grammar."""
    def __init__(self, lm_model_name):
        super(LinearUnigramModel, self).__init__(lm_model_name)
        self.base_grammar = None
        self.primitive_names = None
        self.primitive_name_to_idx = None
        self.idx_to_primitive_name = None 
        self.linear = None
                
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
        Featurizes programs into a unigram probability summary over the DSL of primitives.
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
        # Fits a full cross-product of programs and language
        programs, language = [], []
        for task_programs_and_language in task_to_data.values():
            task_programs, task_language = zip(*task_programs_and_language)
            programs += task_programs
            language += task_language
        unigram_probabilities = self._programs_to_unigram_probabilities(programs, grammar)
        encoded_language = self.language_encoder_fn(language)
        self.input_dim = encoded_language.shape[-1]
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        self._train(encoded_language, unigram_probabilities)
    
    def evaluate_likelihood(self, language, ground_truth_program):
        """Evaluate likelihood of a ground truth program under predicted unigram probabilities."""
        encoded_language = self.language_encoder_fn([language])
        predicted_probabilities = self.linear(encoded_language).detach()[0]
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

def main(args):
    print(f"Running language model evaluations for model: {args['language_encoder']}")
    arc_grammar, language_program_data = load_language_program_data(language_programs_file=LANGUAGE_PROGRAMS_FILE)
    model = get_model(model_tag=args["language_encoder"])
    leave_one_out_evaluation(language_program_data, model, arc_grammar)