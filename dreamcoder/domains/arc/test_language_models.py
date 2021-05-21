"""
test_language_models.py 
Exploratory work setting up language models for the ARC domain.

Usage:
    python bin/arc.py 
        --test_language_models
        --language_encoder t5_linear_predictor # Model name: looks up a model in the model registry.
    
    Baseline results:
    all_programs:
        uniform_unigram_prior : -30.729733065225524
        fitted_unigram_prior, all sentences at once : -19.092839837180854
        t5_linear_predictor, all sentences at once : -22.202822097843782
        t5_mlp_predictor, 100 epochs = -20.480979557565394
        t5_similarity_model, individual sentences: -31.390206467725108
        t5_mixture_model, 100 epochs : -19.152084388523328
    best_programs:
        fitted_unigram_prior: -14.735810399531251
        fitted_bigram_prior: -13.74875295799526
        t5_mlp_predictor, 100 epochs: -16.83427124810492
        t5_mlp_predictor, 100 epochs, batch=10 : -16.252995747965198
        t5_mixture_model, 0.5, 0.5,  100 epochs, batch=10 : -14.847859258665538
        t5_unigram_dc_model, 1000 training steps : -15.522826835452928
        *t5_bigram_dc_model: -13.483465064442436
    
    best_programs, final_dsl:
        fitted_bigram_prior: -13.904265685889866
        t5_bigram_dc_model: -13.264508984184484
        tagged_bigram_model: -13.261363718441588
        t5_tags_bigram_model: -13.132459419297774
        t5_only_pseudotranslation_model = -13.431850043146119
        t5_with_pseudotranslation_model = -13.124230662349463
        t5_train_on_pseudotranslations_test_nothing: -11.788840493275307 
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

from dreamcoder.utilities import lse
from dreamcoder.program import Program
from dreamcoder.grammar import Grammar, ContextualGrammar
from dreamcoder.type import arrow
from dreamcoder.domains.arc.arcPrimitives import *
from dreamcoder.task import Task
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.recognition import GrammarNetwork, LowRank

DATA_DIR = "data/arc"
LANGUAGE_PROGRAMS_FILE = "best_programs_nl_sentences.csv"
PRIMITIVE_NATURAL_LANGUAGE_FILE = "primitiveNamesToDescriptions.json"

# Prediction models supported in this file.
UNIFORM_UNIGRAM_PRIOR = "uniform_unigram_prior" # Uniform distribution over unigrams.
FITTED_UNIGRAM_PRIOR = "fitted_unigram_prior" # Frequency distribution over unigrams.
T5_LINEAR_MODEL = "t5_linear_predictor" # Linear predictor from language-model embeddings to unigrams.
T5_MLP_MODEL = "t5_mlp_predictor" # Linear predictor from language-model embeddings to unigrams.
T5_SIMILARITY_MODEL = "t5_similarity_model" # Normalized similarities between language-model-embeddings and unigram names.
T5_TRAINED_SIMILARITY_MODEL = "t5_trained_similarity_model" # Trains a weight matrix to produce similarities.
T5_MIXTURE_MODEL = "t5_mixture_model" # Mixture between a fitted prior and the unigram predictor
T5_DC_MODEL = "t5_dc_model" # Original bigram grammar predictor in DreamCoder, initialized with the language as a feature extractor.

T5_MODEL = 't5-small' 
ROBERTA_MODEL = 'distilroberta-base'
T5 = 't5'

ARC_REQUEST = arrow(tgridin, tgridout)
VAR_TOKEN = "VAR"
PREDICTION_LOSS = "PREDICTION_LOSS"
LIKELIHOOD_LOSS = "LIKELIHOOD_LOSS"
VERBOSE = True

import logging
import re
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)
            
set_global_logging_level(logging.ERROR) # Suppresses Huggingface

def load_language_program_data(language_programs_file):
    """
    Loads language-program data. Parses programs into Program objects under the ARC DSL.
    Language is generally parsed 
    Returns: 
        arc_grammar: uniform grammar containing the DSL primitives.
        language_program_data: {task_id_{COUNTER}: (language, Program)}.
    """
    # Initialize base primitives to parse the programs.
    basePrimitives()
    leafPrimitives()
    moreSpecificPrimitives()
    arc_grammar = Grammar.uniform(basePrimitives() + leafPrimitives() + moreSpecificPrimitives())
    
    # Load the language data and the parsed programs.
    full_filepath = os.path.join(DATA_DIR, language_programs_file)
    language_program_data = defaultdict(list)
    with open(full_filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "taskName" in row:
                task_name, program, language = row["taskName"], row['program'], row['nlDescription']
                program = Program.parse(program)
                language_program_data[task_name].append((program, language))
            else:
                task_name, program, sentences = load_sentence_data(row)
                if task_name is not None:
                    program = Program.parse(program)
                    for sentence in sentences:
                        language_program_data[task_name].append((program, sentence))
    return arc_grammar, language_program_data 

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
    
def leave_one_out_evaluation(task_to_data, model_class, grammar):
    """Leave one out evaluation on a per-task basis. Fits a model to all of the other tasks, and then computes the average likelihood for the remaining language/program data for the left out task."""
    print(f"Running leave one out evaluation on {len(task_to_data)} tasks.")
    tasks_to_likelihoods = dict()
    likelihoods_language_programs = []
    for idx, task in enumerate(task_to_data):
        heldout_task_programs_and_language  = task_to_data[task]
        training_tasks = {t : d for t, d in task_to_data.items() if t != task}
        model = model_class()
        model.fit(training_tasks, grammar)
        task_likelihoods = []
        for (program, language) in heldout_task_programs_and_language:
            log_likelihood = model.evaluate_likelihood(language, program)
            task_likelihoods.append(log_likelihood)
            likelihoods_language_programs.append((log_likelihood, language, program))
        tasks_to_likelihoods[task] = np.mean(task_likelihoods)
        if idx % 10 == 0:
            print(f"Evaluated: {idx}/{len(task_to_data)}")
            print(f"Current average log likelihood: {np.mean(list(tasks_to_likelihoods.values()))}")
    print(f"Average log likelihood: {np.mean(list(tasks_to_likelihoods.values()))}")
    # Print summary of performance
    for (likelihood, language, program) in sorted(likelihoods_language_programs, key=lambda x : -x[0]):
        print(language)
        print(program, likelihood)
        
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
    def __init__(self, inputs, outputs=None, ground_truth_programs=None):
        self.X = inputs
        self.y = outputs
        self.ground_truth_programs = ground_truth_programs
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        if self.ground_truth_programs:
            return {
                "inputs" : self.X[i],
                "outputs" : self.y[i], 
                "ground_truth" : 
                self.ground_truth_programs[i]
            }
            
        else:
            return self.X[i], self.y[i]

class LanguageModelUnigramModel(nn.Module):
    encoder_cache = defaultdict(lambda: defaultdict())
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
        
        self.base_grammar = None
        self.primitive_names = None
        self.primitive_name_to_idx = None
        self.idx_to_primitive_name = None
        
        self.input_dim = None
        self.output_dim = None
    
    def _initialize_base_unigram_grammar(self, grammar):
        self.base_grammar = grammar
        self.primitive_names = [VAR_TOKEN] + [str(production[-1]) for production in self.base_grammar.productions]
        self.primitive_name_to_idx, self.idx_to_primitive_name = {}, {}
        for (idx, name) in enumerate(self.primitive_names):
            self.primitive_name_to_idx[name] = idx
            self.idx_to_primitive_name[idx] = name
        self.output_dim = len(self.primitive_names)
    
    def _t5_language_encoder_fn(self, language, batching=True):
        """Featurizes a batch of sentences using the T5 model. 
        Each sentence is embedded as a mean over the hidden states of its contextual token embeddings.
        args:
            language: [array of N sentences]
        ret: numpy array of size N x <HIDDEN_STATE_DIM>
        """
        if batching:
            input_ids = self.tokenizer(language, return_tensors="pt", padding=True, truncation=True).input_ids  
            outputs = self.t5_encoder_model(input_ids)
            last_hidden_states = outputs.last_hidden_state 
            reduced = torch.mean(last_hidden_states, dim=1).detach()
        else:
            # Avoids padding, but then manually reduces and combines them.
            reduced_list = []
            for sentence in language:
                if sentence in LanguageModelUnigramModel.encoder_cache[self.lm_model_name]:
                    reduced = LanguageModelUnigramModel.encoder_cache[self.lm_model_name][sentence]
                else:
                    input_ids = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).input_ids  
                    outputs = self.t5_encoder_model(input_ids)
                    last_hidden_states = outputs.last_hidden_state 
                    reduced = torch.mean(last_hidden_states, dim=1).detach()
                    LanguageModelUnigramModel.encoder_cache[self.lm_model_name][sentence] = reduced
                reduced_list.append(reduced)
            reduced = torch.cat(reduced_list, dim=0)
        return reduced
    
    def featuresOfTask(self, task):
        return self.language_encoder_fn([task.language])
        
    def make_unigram_grammar(self, predicted_probabilities, add_variable=False):
        """Builds a unigram grammar over a set of predicted probabilites.
        Normalizes using a softmax function.
        If add_variable: initializes the 'variable' probability to the mean.
        Otherwise: assumes 'variable' probability is in the first place.
        """
        if add_variable:
            avg = torch.mean(predicted_probabilities, 0, keepdim=True)
            predicted_probabilities = torch.cat((avg, predicted_probabilities), -1)
        predicted_probabilities = torch.log(predicted_probabilities)
        unigram_grammar = Grammar(float(predicted_probabilities[0]), 
                       [(float(predicted_probabilities[unigram_idx+1]), t, unigram)
                        for unigram_idx, (_, t, unigram) in enumerate(self.base_grammar.productions)],
                       continuationType=self.base_grammar.continuationType)
        return unigram_grammar
        
    def _program_to_unigram_probabilities(self, program, grammar):
        """Program to normalized unigram probabilities over the grammar. Simplified unigram probabilities, based on unigram counts with a single variable token."""
        unigram_probabilities = np.zeros(len(self.primitive_names))
        unigrams = program.left_order_tokens(show_vars=True)
        for u in unigrams:
            unigram_probabilities[self.primitive_name_to_idx[u]] += 1
        unigram_probabilities /= np.sum(unigram_probabilities)
        # # Add a tiny epsilon.
        # unigram_probabilities += self.epsilon
        # # log_probabilities = np.log(unigram_probabilities)
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

class SimilarityUnigramModel(LanguageModelUnigramModel):
    """Distribution based on similarities between language model sentence embeddings and 
    human-readable unigram names in the base grammar."""
    def __init__(self, lm_model_name, trained=False, primitive_natural_language_file=PRIMITIVE_NATURAL_LANGUAGE_FILE,
    data_dir=DATA_DIR):
        super(SimilarityUnigramModel, self).__init__(lm_model_name)
        self._primitive_natural_language_file = os.path.join(DATA_DIR, primitive_natural_language_file) 
        self._primitive_name_to_human_readable = dict()
        self._primitive_embeddings = None # N_primitives x embedding tensor of embeddings.
        
        self.trained = trained
        
    def _initialize_unigram_name_embeddings(self):
        """Initializes embeddings based on the human readable names for each unigram name."""
        with open(self._primitive_natural_language_file, 'r') as f:
            primitives_to_natural_language = json.load(f)
        
        def get_primitive_to_natural_language(name):
            if name in primitives_to_natural_language:
                if primitives_to_natural_language[name] == None: return ""
                if primitives_to_natural_language[name][-1] == "":
                    return " ".join(primitives_to_natural_language[name][0].split("_"))
                return primitives_to_natural_language[name][-1]
            return ""
        self._primitive_name_to_human_readable = {
            primitive_name : get_primitive_to_natural_language(primitive_name)
            for primitive_name in self.primitive_names
        }
        self._primitive_embeddings = self.language_encoder_fn([self._primitive_name_to_human_readable[primitive_name] for primitive_name in self.primitive_names], batching=False)
        
    def fit(self, task_to_data, grammar):
        """
        task_to_data: dict from task_names to (program, language) for each task.
        """
        self._initialize_base_unigram_grammar(grammar)
        self._initialize_unigram_name_embeddings()
        
        if self.trained:
            self._train(task_to_data, grammar)
    
    def _train(self, task_to_data, grammar):
        programs, language = [], []
        for task_programs_and_language in task_to_data.values():
            task_programs, task_language = zip(*task_programs_and_language)
            programs += task_programs
            language += task_language
        unigram_probabilities = self._programs_to_unigram_probabilities(programs, grammar)
        encoded_language = self.language_encoder_fn(language, batching=False)
        self.input_dim = encoded_language.shape[-1]
        self.R = 16
        self.low_rank = LowRank(self.input_dim, m=1, n=self.output_dim, r=self.R)
        
        self.loss =  nn.BCELoss()
        activation = nn.Tanh
        logits = nn.Sigmoid
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.softmax = nn.Softmax(dim=0)
        self.logits = nn.Sigmoid()
        self.mlp = torch.nn.Sequential(
                            torch.nn.Linear(self.input_dim, self.input_dim),
                            activation(),
                            torch.nn.Linear(self.input_dim, self.input_dim),
                            logits())
        batch_size = 1
        lr_rate = 0.001
        epochs = 200
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_rate, eps=1e-3, amsgrad=True)
        train_dataset = UnigramDataset(encoded_language, unigram_probabilities)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs): 
            losses = []
            for i, (input_batch, output_batch) in enumerate(train_loader):
                input_batch, output_batch = Variable(input_batch), Variable(output_batch)
                self.optimizer.zero_grad()
                predictions = self.forward(input_batch)
                loss = self.loss(torch.unsqueeze(predictions, dim=0), output_batch)
                loss.backward()
                losses.append(loss.data.item())
                self.optimizer.step()
            if VERBOSE: 
                if epoch % 50 == 0:
                    print(f"Epoch: {epoch}, Current average loss: {np.mean(losses)}")
    
    def forward(self, encoded_language):
        """Cosine similarities with a trained low_rank weight matrix.
        """
        weighted_encoding = self.mlp(encoded_language)
        cos_similarities = self.similarity(weighted_encoding, self._primitive_embeddings)
        normalized = self.logits(cos_similarities)
        return normalized
    
    def _similarity_probabilities(self, encoded_language):
        """Predicts probabilities over unigrams based on dot-product similarities to the human-readable
        names of each unigram."""
        similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_similarities = similarity(encoded_language, self._primitive_embeddings)
        # Normalize into log probabilities.
        normalized = nn.LogSoftmax(dim=0)(cos_similarities)
        return normalized
    
    def evaluate_likelihood(self, language, ground_truth_program):
        """Evaluate likelihood of a ground truth program under predicted unigram probabilities."""
        encoded_language = self.language_encoder_fn([language]).detach()
        if self.trained:
            predicted_probabilities = self.forward(encoded_language)
        else:
            predicted_probabilities = self._similarity_probabilities(encoded_language)
        unigram_grammar = self.make_unigram_grammar(predicted_probabilities)
        
        ll = unigram_grammar.logLikelihood(request=ARC_REQUEST, expression=ground_truth_program)
        return ll
        
@register_model(T5_SIMILARITY_MODEL)
class T5SimilarityUnigramModel(SimilarityUnigramModel):
    def __init__(self):
        super(T5SimilarityUnigramModel, self).__init__(lm_model_name=T5_MODEL) 

@register_model(T5_TRAINED_SIMILARITY_MODEL)
class T5TrainedSimilarityUnigramModel(SimilarityUnigramModel):
    def __init__(self):
        super(T5TrainedSimilarityUnigramModel, self).__init__(lm_model_name=T5_MODEL, trained=True)
        
class LanguageModelClassifierUnigramModel(LanguageModelUnigramModel):
    """Linear classifier from language model sentence embeddings to unigrams in the base grammar."""
    def __init__(self, lm_model_name, n_hidden_layers=0, prior_mixture=False, loss_function_type=PREDICTION_LOSS):
        super(LanguageModelClassifierUnigramModel, self).__init__(lm_model_name) 
        self.n_hidden_layers = n_hidden_layers
        self.classifier = None
        self.loss_function_type = loss_function_type
        self.loss = self.get_loss_fn(loss_function_type)
        self.epsilon = 0.01
        
        self.prior_mixture = prior_mixture
        self.fitted_prior = None
        self.prior_weight = 0.1
        self.learned_weight = 0.9
    
    def get_loss_fn(self, loss_function_type):
        self.prediction_loss_fn = nn.BCELoss()
        if loss_function_type == PREDICTION_LOSS:
            return self.prediction_loss
        elif loss_function_type == LIKELIHOOD_LOSS:
            return self.likelihood_loss_fn
        else:
            assert False
            
    def prediction_loss(self, predictions, outputs, programs=None, normalize=False):
        """BCE logits loss between unnormalized probabilities."""
        return self.prediction_loss_fn(predictions, outputs)
        
    def likelihood_loss_fn(self, prediction, output, ground_truth_program=None):
        unigram_grammar = self.make_unigram_grammar(prediction[0])
        ll = unigram_grammar.logLikelihood(request=ARC_REQUEST, expression=ground_truth_program)
        al = self.prediction_loss_fn(prediction, output)
        return al

    def _train(self, inputs, outputs, ground_truth_programs, epochs=200):
        batch_size = 10
        lr_rate = 0.001
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_rate, eps=1e-3, amsgrad=True)

        train_dataset = UnigramDataset(inputs, outputs)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs): 
            losses = []
            for i, (input_batch, output_batch) in enumerate(train_loader):
                input_batch, output_batch = Variable(input_batch), Variable(output_batch)
                self.optimizer.zero_grad()
                predictions = self.classifier(input_batch)
                loss = self.loss(predictions, output_batch)
                loss.backward()
                losses.append(loss.data.item())
                self.optimizer.step()
            if VERBOSE: 
                if epoch % 50 == 0:
                    print(f"Epoch: {epoch}, Current average loss: {np.mean(losses)}")
        
    def _initialize_classifier(self):
        self.hidden_size = self.input_dim
        if self.n_hidden_layers == 0:
            self.classifier = nn.Linear(self.input_dim, self.output_dim)
        elif self.n_hidden_layers == 1:
            activation = nn.Tanh
            logits = nn.Sigmoid
            self.classifier = torch.nn.Sequential(
                                torch.nn.Linear(self.input_dim, self.hidden_size),
                                activation(),
                                torch.nn.Linear(self.hidden_size, self.output_dim),
                                logits())
        else:
            print("Not implemented")
            assert False
    
    def fit(self, task_to_data, grammar):
        """
        task_to_data: dict from task_names to (program, language) for each task.
        """
        self._initialize_base_unigram_grammar(grammar)
        if self.prior_mixture:
            self.fitted_prior = FittedUnigramPriorModel()
            self.fitted_prior.fit(task_to_data, grammar)
        # Fits a full cross-product of programs and language
        programs, language = [], []
        for task_programs_and_language in task_to_data.values():
            task_programs, task_language = zip(*task_programs_and_language)
            programs += task_programs
            language += task_language
        unigram_probabilities = self._programs_to_unigram_probabilities(programs, grammar)
        encoded_language = self.language_encoder_fn(language, batching=False)
        self.input_dim = encoded_language.shape[-1]
        self._initialize_classifier()
        self._train(encoded_language, unigram_probabilities, programs)
    
    def get_prior_mixture(self, predicted_probabilities):
        prior_log_probabilities = [self.fitted_prior.prior_grammar.logVariable] + [p[0] for p in self.fitted_prior.prior_grammar.productions]
        prior_probabilities = np.exp(prior_log_probabilities)
        
        mixture_probabilities = torch.tensor((self.prior_weight * prior_probabilities)) + (self.learned_weight * predicted_probabilities)
        return mixture_probabilities
        
    def evaluate_likelihood(self, language, ground_truth_program):
        """Evaluate likelihood of a ground truth program under predicted unigram probabilities."""
        encoded_language = self.language_encoder_fn([language])
        predicted_probabilities = self.classifier(encoded_language).detach()[0]
        
        if self.prior_mixture:
            predicted_probabilities = self.get_prior_mixture(predicted_probabilities)
        unigram_grammar = self.make_unigram_grammar(predicted_probabilities)
        ll = unigram_grammar.logLikelihood(request=ARC_REQUEST, expression=ground_truth_program)
        return ll

@register_model(T5_LINEAR_MODEL)
class T5LinearUnigramModel(LanguageModelClassifierUnigramModel):
    def __init__(self):
        super(T5LinearUnigramModel, self).__init__(lm_model_name=T5_MODEL, n_hidden_layers=0)

@register_model(T5_MLP_MODEL)
class T5MLPUnigramModel(LanguageModelClassifierUnigramModel):
    def __init__(self):
        super(T5MLPUnigramModel, self).__init__(lm_model_name=T5_MODEL, n_hidden_layers=1)

@register_model(T5_MIXTURE_MODEL)
class T5MixtureUnigramModel(LanguageModelClassifierUnigramModel):
    def __init__(self):
        super(T5MixtureUnigramModel, self).__init__(lm_model_name=T5_MODEL, n_hidden_layers=1, prior_mixture=True)


def main(args):
    print(f"Running language model evaluations for model: {args['language_encoder']}")
    arc_grammar, language_program_data = load_language_program_data(language_programs_file=LANGUAGE_PROGRAMS_FILE)
    model = get_model(model_tag=args["language_encoder"])
    leave_one_out_evaluation(language_program_data, model, arc_grammar)