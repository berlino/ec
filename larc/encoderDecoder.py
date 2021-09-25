import os
import pickle
import random
import time
import torch
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchviz import make_dot

import torch
torch.manual_seed(2)
random.seed(0)

from dreamcoder.domains.arc.main import retrieveARCJSONTasks
from dreamcoder.domains.arc.arcPrimitives import basePrimitives, leafPrimitives, moreSpecificPrimitives, tgridin, tgridout
from dreamcoder.domains.arc.utilsPostProcessing import resume_from_path
from dreamcoder.grammar import Grammar
from dreamcoder.program import Program
from dreamcoder.recognition import RecognitionModel
from dreamcoder.type import arrow
from dreamcoder.utilities import ParseFailure, EarlyStopping, parallelMap

from larc.decoder import *
from larc.decoderUtils import *
from larc.encoder import LARCEncoder
from larc.larcDataset import *
from larc.train import *

class EncoderDecoder(nn.Module):
    """
    Program Synthesis model that takes in a feature vector (encoding) from which it outputs (decodes) a program. 

    Encoder:
        1. Encode IO examples with existing ARC CNN
        2. Encode NL description with existing T5 encoder

    Decoder:
        1. Use encoder output + partial program encoding as attention query, and NL description words / BERT encodings as keys and values
        2. Concatenate output from 1 with query from 1
        3. Use output from 2 as query and token embeddings as keys to get attention vector
    """

    def __init__(self, grammar, request, cuda, device, program_embedding_size=128, program_size=128, primitive_to_idx=None):
        super().__init__()
        
        self.device = device
        # there are three additional tokens, one for the start token input grid (tgridin) variable and the other for input 
        # variable of lambda expression (assumes no nested lambdas)
        print("Starting to load decoder")
        self.decoder = Decoder(embedding_size=program_embedding_size, grammar=grammar, request=request, cuda=cuda, device=device, max_program_length=MAX_PROGRAM_LENGTH, 
            encoderOutputSize=64, primitive_to_idx=primitive_to_idx)
        print("Finished loading Decoder, starting to load Encoder")
        self.encoder = LARCEncoder(cuda=cuda, device=device)
        print("Finished loading Encoder")
    
        if cuda: self.cuda()

    def forward(self, io_grids, test_in, desc_tokens, mode, targets=None):

        encoderOutputs = self.encoder(io_grids, test_in, desc_tokens)
        print("encoder Outputs size: {}".format(encoderOutputs.size()))

        batch_scores = decode_score(self.decoder, encoderOutputs, targets, self.device)
        return batch_scores


        # batch_scores = torch.empty(encoderOutputs.size(1), requires_grad=False, device=self.device)
        # programs = []
        # for i in range(encoderOutputs.size(1)):
        #     program = decode_single(self.decoder, encoderOutputs[:, i], targets[i, :], mode)
        #     batch_scores[i] = program.totalScore
        #     programs.append(program)
        # return programs, batch_scores

def main():

    use_cuda = False
    batch_size = 16
    lr = 0.001
    weight_decay = 0.0
    beta = 0.0
    epochs_per_experience_replay = 1
    beam_width = 32
    epsilon = 0.3
    n = 32

    if use_cuda: 
        assert torch.cuda.is_available()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print("Using {}".format(device)) 
    # # load grammar / DSL
    primitives_to_use = "base"
    if primitives_to_use == "base":
        primitives = basePrimitives() + leafPrimitives()
    else:
        primitives = basePrimitives() + leafPrimitives() + moreSpecificPrimitives()
    grammar = Grammar.uniform(primitives)

    # create dict from tokens to corresponding indices (to be used as indices for nn.Embedding)
    token_to_idx = {"START": 0, "LAMBDA": 1, "LAMBDA_INPUT":2, "INPUT": 3}
    num_special_tokens = len(token_to_idx)
    token_to_idx.update({str(token):i+num_special_tokens for i,token in enumerate(grammar.primitives)})
    idx_to_token = {idx: token for token,idx in token_to_idx.items()}
  
    # load model
    request = arrow(tgridin, tgridout)
    print("Startin to load model")
    model = EncoderDecoder(grammar=grammar, request=request, cuda=use_cuda, device=device, program_embedding_size=128, program_size=128, primitive_to_idx=token_to_idx)
    # model.share_memory()
    # model.load_state_dict(torch.load("data/larc/imitation_learning_model.pt")["model_state_dict"])
    print("Finished loading model")

    # load tasks to check sampled programs against
    dataDirectory = "arc_data/data/"
    tasks = retrieveARCJSONTasks(dataDirectory + 'training', useEvalExamplesForTraining=False, filenames=None)

    # load already discovered programs
    tasks_dir = "data/larc/tasks_json"
    json_file_name = "data/arc/prior_enumeration_frontiers_8hr.json"
    task_to_programs_json = json.load(open(json_file_name, 'r'))
    task_to_programs = load_task_to_programs_from_frontiers_json(grammar, token_to_idx, max_program_length=MAX_PROGRAM_LENGTH, 
        task_to_programs_json=task_to_programs_json, device=device, token_pad_value=len(idx_to_token))
    print("Loaded task to programs")

    # tasks_with_programs = [t for t,programs in task_to_programs.items() if len(programs) > 0]
    # train_task_names, test_task_names = next(getKfoldSplit(tasks_with_programs, 0.8, 5))

    # load dataset for torch model
    larc_train_dataset = LARC_Cell_Dataset(tasks_dir, tasks_subset=None, num_ios=MAX_NUM_IOS, resize=(30, 30), for_synthesis=True, 
        beta=beta, task_to_programs=None, device=device)
    print("Finished loading dataset ({} samples)".format(len(larc_train_dataset))) 

    # data_loader = DataLoader(larc_train_dataset[0:4], batch_size=batch_size, collate_fn =lambda x: collate(x, False), drop_last=True)
    # print("Finished loading DataLoaders")
    # imitation learning
    # model, epoch_train_scores, test_scores = train_imitiation_learning(model, data_loader, test_loader=None, batch_size=1, 
    #   lr=lr, weight_decay=weight_decay, num_epochs=10, earlyStopping=False)
 
    for iteration in range(2):

        if iteration == 0:
            task_to_programs = {k:v for k,v in task_to_programs.items() if len(v) > 0}
            task_to_correct_programs = task_to_programs
            print(len(task_to_programs))
        
        if len(task_to_correct_programs) > 0:
            model = train_experience_replay(model, task_to_correct_programs, tasks_dir=tasks_dir, beta=beta, num_epochs=epochs_per_experience_replay, lr=lr, weight_decay=weight_decay, batch_size=batch_size, device=device)
            
            start_time = time.time()
            model = model.to(device=torch.device("cpu"))
            print("{}s to transfer model to CPU".format(time.time() - start_time))
            torch.save({
               'model_state_dict': model.state_dict(),
            }, 'data/larc/model_{}.pt'.format(iteration))
        else:
            print("No correct programs found on iteration {}".format(iteration))

        
        start_time = time.time()
        # decode with randomized beam search
        task_to_decoded_programs, task_to_lls = decode(model, grammar, larc_train_dataset, tasks, batch_size, how="randomized_beam_search", n=n, beam_width=beam_width, epsilon=epsilon)
        print("\nFinished Decoding in {}s \n".format(time.time() - start_time))
        
        # experience replay train with discovered program
        task_to_correct_programs = {}
        print("Discovered below programs:\n")
        for task,log_likelihoods in task_to_lls.items():
            for i,ll in log_likelihoods:
                if ll == 0.0:
                    res = task_to_correct_programs.get(task, [])
                    program = task_to_decoded_programs[task][i]
                    programTokenSeq, programWeight = program.programTokenSeq, program.totalScore[0]
                    print("Task {}: {}".format(task, programStringsSeq))
                    res.append((programTokenSeq, programWeight))
                    task_to_correct_programs[task] = res
        print(task_to_correct_programs)
