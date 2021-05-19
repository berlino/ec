"""
language_model_feature_extractor.py | Feature extractors initialized with large scale language models.
"""
T5_MODEL = 't5-small' 
T5 = 't5'

from collections import defaultdict

import torch
import torch.nn as nn
from transformers import pipeline, T5Tokenizer, T5EncoderModel

import logging
import re
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)
            
set_global_logging_level(logging.ERROR) # Suppresses Huggingface

class LMFeatureExtractor(nn.Module):
    encoder_cache = defaultdict(lambda: defaultdict())
    
    def __init__(self, tasks=[], testingTasks=[], cuda=False, lm_model_name=T5_MODEL):
        super(LMFeatureExtractor, self).__init__()
        self.CUDA = cuda
        self.recomputeTasks = True
        self.outputDimensionality = 512
        
        self.lm_model_name = lm_model_name
        if T5 in self.lm_model_name:
            self.tokenizer = T5Tokenizer.from_pretrained(self.lm_model_name)
            self.t5_encoder_model = T5EncoderModel.from_pretrained(self.lm_model_name)
            self.language_encoder_fn = self._t5_language_encoder_fn
        else:
            self.language_encoder_fn = pipeline('feature-extraction', self.lm_model_name)
    
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
    
    def forward(self, sentences):
        return self.language_encoder_fn(sentences)
    
    def featuresOfTask(self, t):
        return self(t.sentences)
            
    def taskOfProgram(self, p, tp):
        """
        For simplicitly we only use one example per task randomly sampled from
        all possible input grids we've seen.
        """
        def randomInput(t): return random.choice(self.argumentsWithType[t])

        startTime = time.time()
        examples = []
        while True:
            # TIMEOUT! this must not be a very good program
            if time.time() - startTime > self.helmholtzTimeout: return None

            # Grab some random inputs
            xs = [randomInput(t) for t in tp.functionArguments()]
            try:
                y = runWithTimeout(lambda: p.runWithArguments(xs), self.helmholtzEvaluationTimeout)
                examples.append((tuple(xs),y))
                if len(examples) >= 1:
                    return Task("Helmholtz", tp, examples)
            except: continue
        return None

    def featuresOfTasks(self, ts, t2=None):  # Take a task and returns [features]
        """Takes the goal first; optionally also takes the current state second"""
        return [self.featuresOfTask(t) for t in ts]