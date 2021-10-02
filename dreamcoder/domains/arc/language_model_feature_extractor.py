"""
language_model_feature_extractor.py | Feature extractors initialized with large scale language models.
"""
T5_MODEL = 't5-small' 
T5 = 't5'
UNK = ""

from collections import defaultdict

import csv
import json
import random
import string

import torch
import torch.nn as nn
from transformers import pipeline, T5Tokenizer, T5EncoderModel

from dreamcoder.task import Task
from dreamcoder.domains.arc.cnn_feature_extractor import ArcCNN

import logging
import re
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)
            
set_global_logging_level(logging.ERROR) # Suppresses Huggingface

def normalize_sentences(sentences):
    return [normalize_sentence(sentence) for sentence in sentences]

def normalize_sentence(sentence):
    sentence = " ".join(sentence.lower().split())
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    return sentence
    
class LMFeatureExtractor(nn.Module):
    encoder_cache = defaultdict(lambda: defaultdict())
    special = 'arc'
    
    def __init__(self, tasks=[], testingTasks=[], cuda=False, lm_model_name=T5_MODEL, use_language_model=True, tagged_annotations_file=None,
    additional_feature_file=None,
    primitive_names_to_descriptions=None,
    pseudo_translation_probability=0.0,
    should_normalize=True,
    use_cnn=False):
        super(LMFeatureExtractor, self).__init__()
        self.CUDA = cuda
        self.recomputeTasks = True
        self.epsilon = 0.001

        self.lm_model_name = lm_model_name
        if T5 in self.lm_model_name:
            self.tokenizer = T5Tokenizer.from_pretrained(self.lm_model_name, cache_dir=".cache/")
            self.t5_encoder_model = T5EncoderModel.from_pretrained(self.lm_model_name, cache_dir=".cache/")
            self.language_encoder_fn = self._t5_language_encoder_fn
        else:
            self.language_encoder_fn = pipeline('feature-extraction', self.lm_model_name)
        
        self.should_normalize = should_normalize
        self.use_language_model = use_language_model
        self.use_tagged_features = (tagged_annotations_file is not None) or (additional_feature_file is not None)
        self.use_primitive_names_to_descriptions = (primitive_names_to_descriptions is not None)
        
        if self.use_tagged_features:
            if tagged_annotations_file is not None:
                self.tagged_features_data = self._initialize_tagged_features(tagged_annotations_file)
            if additional_feature_file is not None:
                self.tagged_features_data = self._initialize_json_features(additional_feature_file)
        
        if self.use_primitive_names_to_descriptions:
            self.primitive_names_to_descriptions = self._initialize_primitive_names_to_descriptions(primitive_names_to_descriptions)
            self.pseudo_translation_probability = pseudo_translation_probability
        
        self.use_cnn = use_cnn
        if self.use_cnn:
            self.cnn_H = 64
            self.arc_cnn = ArcCNN(tasks, testingTasks, cuda)
        
        self.outputDimensionality = self._compute_output_dimensionality()
    
    def _initialize_primitive_names_to_descriptions(self, primitive_names_to_descriptions_file):
        """Initializes a primitive_names_to_descriptions dictionary."""
        with open(primitive_names_to_descriptions_file) as f:
            raw = json.load(f)
        primitive_names_to_descriptions = defaultdict(str)
        for name in raw:
            if raw[name] is None:
                continue
            primitive_names_to_descriptions[name] = raw[name][-1]
        return primitive_names_to_descriptions
    
    def _initialize_json_features(self, json_features_file):
        """Initalizes self.tasks_to_tagged_features from a JSON file that annotates tasks with a feature vector.
        Returns defaultdict {task_id : torch_feature_vector} and sets self.tagged_feature_output_dim. Unfound tasks will return a zero vector of the appropriate size.
        """
        self.tasks_to_features_raw = defaultdict(list)
        
        sample_feature_vector = None
        with open(json_features_file) as f:
            raw_json_features = json.load(f)["task_to_feature_vector"]
            for task_name in raw_json_features:
                self.tagged_feature_output_dim = len(raw_json_features[task_name])
                task_feature_vector = torch.as_tensor(raw_json_features[task_name], dtype=torch.float32)
                sample_feature_vector = task_feature_vector
                self.tasks_to_features_raw[task_name] = task_feature_vector
        
        self.tasks_to_tagged_features = defaultdict(lambda : torch.zeros_like(sample_feature_vector) + self.epsilon)
        for task in self.tasks_to_features_raw:
            self.tasks_to_tagged_features[task] = self.tasks_to_features_raw[task]
    
    def _initialize_tagged_features(self, tagged_annotations_file):
        """Initalizes self.tasks_to_tagged_features from a CSV file that annotates tasks with a feature vector.
        Returns defaultdict {task_id : torch_feature_vector} and sets self.tagged_feature_output_dim. Unfound tasks will return a zero vector of the appropriate size.
        """
        self.tag_keys = None
        self.tasks_to_features_raw = defaultdict(list)
        
        def to_int(tag_value_string):
            if len(tag_value_string) < 1:
                return 0
            else:
                return int(tag_value_string)
                
        sample_feature_vector = None
        with open(tagged_annotations_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.tag_keys is None:
                    self.tag_keys = sorted([k for k in row.keys() if k.startswith("tag")])
                task_name, phrase_kind = row['task_name'], row['phrase_kind']
                if phrase_kind == 'output':
                    task_feature_vector = torch.as_tensor([to_int(row[tag]) for tag in self.tag_keys], dtype=torch.float32)
                    self.tasks_to_features_raw[task_name].append(task_feature_vector)
                    sample_feature_vector = task_feature_vector
        self.tagged_feature_output_dim = len(self.tag_keys)
        self.tasks_to_tagged_features = defaultdict(lambda : torch.zeros_like(sample_feature_vector) + self.epsilon)
        for task in self.tasks_to_features_raw:
            self.tasks_to_tagged_features[task] = torch.mean(torch.stack(self.tasks_to_features_raw[task]),dim=0) + self.epsilon
    
    def _compute_output_dimensionality(self):
        outputDimensionality = 0
        if self.use_language_model or self.use_primitive_names_to_descriptions:
            if self.lm_model_name == T5_MODEL:
                outputDimensionality += 512
            else:
                assert False
        if self.use_tagged_features:
            outputDimensionality += self.tagged_feature_output_dim
        if self.use_cnn:
            outputDimensionality += self.arc_cnn.outputDimensionality
        assert outputDimensionality > 0
        return outputDimensionality
    
    def _t5_language_encoder_fn(self, language, batching=False):
        """Featurizes a batch of sentences using the T5 model. 
        Each sentence is embedded as a mean over the hidden states of its contextual token embeddings.
        Reduces again over the sentences
        args:
            language: [array of N sentences]
        ret: 1 x <HIDDEN_STATE_DIM>
        """
        if batching:
            input_ids = self.tokenizer(language, return_tensors="pt", padding=True, truncation=True).input_ids  
            outputs = self.t5_encoder_model(input_ids)
            last_hidden_states = outputs.last_hidden_state 
            reduced = torch.mean(last_hidden_states, dim=1).detach()
        else:
            # Avoids padding, but then manually reduces and combines them.
            reduced_list = []
            if len(language) < 1:
                language = [UNK]
            for sentence in language:
                if sentence in LMFeatureExtractor.encoder_cache[self.lm_model_name]:
                    reduced = LMFeatureExtractor.encoder_cache[self.lm_model_name][sentence]
                else:
                    input_ids = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).input_ids  
                    outputs = self.t5_encoder_model(input_ids)
                    last_hidden_states = outputs.last_hidden_state 
                    reduced = torch.mean(last_hidden_states, dim=1).detach()
                    LMFeatureExtractor.encoder_cache[self.lm_model_name][sentence] = reduced
                reduced_list.append(reduced)
            reduced = torch.cat(reduced_list, dim=0)
        all_sentences_reduced = torch.mean(reduced, dim=0)
        return all_sentences_reduced
    
    def forward(self, task, task_name, sentences):
        task_name = self.original_task_name(task_name)
        features = []
        if self.should_normalize:
            language = normalize_sentences(sentences)
        if self.use_primitive_names_to_descriptions:
            pseudo_translations = self.get_pseudo_translations(task)
            language = self.get_pseudo_translations_or_sentences(pseudo_translations, sentences)
            features.append(self.language_encoder_fn(language))
        if self.use_language_model:
            features.append(self.language_encoder_fn(sentences))
        if self.use_tagged_features:
            features.append(self.tasks_to_tagged_features[task_name])
        if self.use_cnn:
            features.append(self.arc_cnn.featuresOfTask(task))
        return torch.cat(features)
    
    def get_pseudo_translations_or_sentences(self, pseudo_translations, sentences):
        if not self.train:
            return sentences
        should_use_pseudo = random.random() < self.pseudo_translation_probability
        if len(sentences) < 1:
            should_use_pseudo = True
        return pseudo_translations if should_use_pseudo else sentences
    
    def get_pseudo_translations(self, task):
        pseudo_translations = [UNK]
        def pseudo_translation_function(program_tokens):
            pseudo_translation = " ".join(self.primitive_names_to_descriptions.get(t, "").strip() for t in program_tokens)
            pseudo_translation = " ".join(pseudo_translation.split())
            return pseudo_translation
        pseudo_translations = [pseudo_translation_function(p.uncurry().left_order_tokens()) for p in task.solution_programs]
        return pseudo_translations
    
    def original_task_name(self, task_name):
        """Utility function: we sometimes append _tags to the end of task names."""
        return task_name.split("_")[0]
    
    def featuresOfTask(self, t):
        return self(t, t.name, t.sentences)
            
    def taskOfProgram(self, p, tp):
        """
        Degenerate task with no examples.
        """
        examples = []
        task = Task("Helmholtz", tp, examples)
        task.sentences = []
        return task

    def featuresOfTasks(self, ts, t2=None):  # Take a task and returns [features]
        """Takes the goal first; optionally also takes the current state second"""
        return [self.featuresOfTask(t) for t in ts]
