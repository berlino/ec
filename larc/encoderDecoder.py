import os
import pickle
import random
import torch
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
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
from dreamcoder.utilities import ParseFailure, EarlyStopping

from larc.decoder import *
from larc.decoderUtils import *
from larc.encoder import LARCEncoder
from larc.larcDataset import *

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

    def __init__(self, batch_size, grammar, request, cuda, device, program_embedding_size=128, program_size=128, primitive_to_idx=None):
        super().__init__()
        
        self.device = device
        self.batch_size = batch_size
        self.encoder = LARCEncoder(cuda=cuda, device=device)
        # there are three additional tokens, one for the start token input grid (tgridin) variable and the other for input 
        # variable of lambda expression (assumes no nested lambdas)
        self.decoder = Decoder(embedding_size=program_embedding_size, batch_size=batch_size, grammar=grammar, request=request, cuda=cuda, device=device, max_program_length=MAX_PROGRAM_LENGTH, 
            encoderOutputSize=64, primitive_to_idx=primitive_to_idx)
        
        if cuda: self.cuda()

    def forward(self, io_grids, test_in, desc_tokens, mode, targets=None):

        encoderOutputs = self.encoder(io_grids, test_in, desc_tokens)

        scores = torch.empty(self.batch_size, device=self.device)
        for i in range(self.batch_size):
            program_string, score = sample_decode(self.decoder, encoderOutputs[:, i], mode, targets[i, :])
            scores[i] = score
        return program_string, scores

def train_imitiation_learning(model, train_dataset, test_dataset, batch_size, lr, weight_decay, num_epochs, earlyStopping=True):

    model.train()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=lambda x: collate(x, True), drop_last=True)
    # we use a batch size equal to the test dataset so as not to drop any samples
    test_batch_size = len(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, collate_fn=lambda x: collate(x, True), drop_last=True)

    epoch_train_scores = []
    test_scores = []

    n_epochs_stop = 10
    if earlyStopping:
        ep = EarlyStopping(patience=10, n_epochs_stop=n_epochs_stop, init_best_val_loss=float('INF'))
        assert len(test_dataset) > 0
    
    # Imitation learning training
    for epoch in range(num_epochs):
        
        epoch_score = 0.0

        for batch in train_loader:
            # the sequence will always be the ground truth since we run forward in "score" mode
            token_sequences, scores = model(io_grids=batch["io_grids"], test_in=batch["test_in"], desc_tokens=batch["desc_tokens"], mode="score", targets=batch['programs'])
            
            batch_score = - (torch.sum(scores) / batch_size)
            epoch_score += batch_score

            batch_score.backward()
            optimizer.step()
            optimizer.zero_grad()

        epoch_score = epoch_score / len(train_loader)
        epoch_train_scores.append(epoch_score)
        print("Training score at epoch {}: {}".format(epoch, epoch_score))


        # Get test set performance
        if epoch % 5 == 0:

            test_score = 0.0
            num_batches = 0
            for batch in test_loader:
                # the sequence will always be the ground truth since we run forward in "score" mode
                token_sequences, scores = model(io_grids=batch["io_grids"], test_in=batch["test_in"], desc_tokens=batch["desc_tokens"], mode="score", targets=batch['programs'])
                
                batch_score = - (torch.sum(scores) / test_batch_size)
                test_score += batch_score
                num_batches += 1

            epoch_test_score = test_score / num_batches
            print("Test score at epoch {}: {}".format(epoch, epoch_test_score))
            
            test_scores.append(test_score / num_batches)

            if earlyStopping:
                shouldStop, bestModel = ep.should_stop(epoch, epoch_test_score, model)
                if shouldStop:
                    print("Holdout loss stopped decreasing after {} epochs".format(n_epochs_stop))
                    return bestModel, epoch_train_scores, test_scores

    return model, epoch_train_scores, test_scores

def getKfoldSplit(taskNames, trainRatio, k):
    
    totalNumTasks = len(set(taskNames))
    numTrain = int(trainRatio * totalNumTasks)

    for i in range(k):
        trainTaskNames = random.sample(taskNames, numTrain)
        testTaskNames = list(set(taskNames).difference(trainTaskNames))

        yield trainTaskNames, testTaskNames


def main():

    use_cuda = True
    batch_size = 1

    if use_cuda: 
        assert torch.cuda.is_available()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
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

    request = arrow(tgridin, tgridout)
    model = EncoderDecoder(batch_size=batch_size, grammar=grammar, request=request, cuda=use_cuda, device=device, program_embedding_size=128, program_size=128, primitive_to_idx=token_to_idx)

    # # load dataset
    tasks_dir = "data/larc/tasks_json"
    json_file_name = "data/arc/prior_enumeration_frontiers_8hr.json"
    task_to_programs_json = json.load(open(json_file_name, 'r'))
    task_to_programs = load_task_to_programs_from_frontiers_json(grammar, token_to_idx, max_program_length=MAX_PROGRAM_LENGTH, task_to_programs_json=task_to_programs_json)
    
    tasks_with_programs = [t for t,programs in task_to_programs.items() if len(programs) > 0]
    train_task_names, test_task_names = next(getKfoldSplit(tasks_with_programs, 0.8, 5))
    
    print(train_task_names, len(train_task_names))
    print(test_task_names, len(test_task_names))

    larc_train_dataset = LARC_Cell_Dataset(tasks_dir, tasks_subset=train_task_names, num_ios=MAX_NUM_IOS, resize=(30, 30), for_synthesis=True, task_to_programs=task_to_programs, device=device)
    # larc_test_dataset = LARC_Cell_Dataset(tasks_dir, tasks_subset=test_task_names, num_ios=MAX_NUM_IOS, resize=(30, 30), task_to_programs=task_to_programs, device=device)

    print("Total train samples: {}".format(len(larc_train_dataset)))
    # print("Total test samples: {}".format(len(larc_test_dataset)))
 
    # model = train_imitiation_learning(model, larc_train_dataset, larc_test_dataset, batch_size=batch_size, lr=1e-3, weight_decay=0.0, num_epochs=3)
    model.load_state_dict(torch.load("model.pt")["model_state_dict"])
    
    data_loader = DataLoader(larc_train_dataset[0:5], batch_size=batch_size, collate_fn =lambda x: collate(x, False), drop_last=True)
    task_to_programs_sampled = decode(model, data_loader, batch_size, how="randomized_beam_search", n=20)
    print("\nFinished Decoding\n")
    print("resulting data structure: ", task_to_programs_sampled)
    

    # run sampled programs with ocaml
    homeDirectory = "/".join(os.path.abspath(__file__).split("/")[:-4])
    dataDirectory = "arc_data/data/"
    tasks = retrieveARCJSONTasks(dataDirectory + 'training', useEvalExamplesForTraining=False, filenames=None)
    # getting actual Task objects instead of just task_name (string)
    train_tasks = [t for t in tasks if t.name in task_to_programs_sampled]
    task_to_log_likelihoods = execute_programs(train_tasks, grammar, task_to_programs_sampled)
    for item in task_to_log_likelihoods:
        print(item["task"], item["log_likelihoods"])
        print("----------------------------------------------------------")


