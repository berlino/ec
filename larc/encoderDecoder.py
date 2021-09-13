import os
import pickle
import torch
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchviz import make_dot

from dreamcoder.domains.arc.main import retrieveARCJSONTasks
from dreamcoder.domains.arc.arcPrimitives import basePrimitives, leafPrimitives, moreSpecificPrimitives, tgridin, tgridout
from dreamcoder.domains.arc.utilsPostProcessing import resume_from_path
from dreamcoder.grammar import Grammar
from dreamcoder.program import Program
from dreamcoder.recognition import RecognitionModel
from dreamcoder.type import arrow
from dreamcoder.utilities import ParseFailure

from larc.decoder import *
from larc.decoderUtils import *
from larc.encoder import LARCEncoder
from larc.larcDataset import *

MAX_NUM_IOS = 3

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

        token_sequences = []
        scores = torch.empty(self.batch_size, device=self.device)
        for i in range(self.batch_size):
            token_sequence, program_string, score = self.decoder(encoderOutputs[:, i], mode, targets[i, :])
            token_sequences.append(token_sequences)
            scores[i] = score
        return token_sequences, program_string, scores

def collate(x):

    def stack_entry(x, name):
        return torch.stack([x[i][name] for i in range(len(x))])

    # stack all tensors of the same input/output type and the same example index to form batch
    io_grids_batched = [(torch.stack([x[i]["io_grids"][ex_idx][0] for i in range(len(x))]), torch.stack([x[i]["io_grids"][ex_idx][1] for i in range(len(x))])) 
        for ex_idx in range(MAX_NUM_IOS)]

    return {"io_grids": io_grids_batched,
            "test_in": stack_entry(x, "test_in"), 
            "desc_tokens": {key: torch.stack([x[i]["desc_tokens"][key] for i in range(len(x))]) for key in x[0]["desc_tokens"].keys()},
            "programs": stack_entry(x, "programs")}

def train_imitiation_learning(model, tasks, batch_size, lr, weight_decay, num_epochs):

    model.train()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)

    train_loader = DataLoader(tasks, batch_size=batch_size, collate_fn=collate, drop_last=True)

    epoch_scores = []
    
    # Imitation learning training
    for epoch in range(num_epochs):
        
        epoch_score = 0.0

        for batch in train_loader:
            # the sequence will always be the ground truth since we run forward in "score" mode
            token_sequences, scores = model(io_grids=batch["io_grids"], test_in=batch["test_in"], desc_tokens=batch["desc_tokens"], mode="score", targets=batch['programs'])
            
            batch_score = torch.sum(scores) / batch_size
            epoch_score += batch_score

            (-batch_score).backward()
            optimizer.step()
            optimizer.zero_grad()

        epoch_score = epoch_score / len(train_loader)
        epoch_scores.append(epoch_score)
        print("Score at epoch {}: {}".format(epoch, epoch_score))

    torch.save({
        'num_epochs': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'model.pt')

    return model


def main():

    use_cuda = False
    batch_size = 2

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
    larc_train_dataset = LARC_Cell_Dataset(tasks_dir, tasks_subset=None, num_ios=MAX_NUM_IOS, resize=(30, 30), task_to_programs=task_to_programs, device=device)
    dataset = larc_train_dataset[0:8]
 
    # model = train_imitiation_learning(model, dataset, batch_size=batch_size, lr=1e-3, weight_decay=0.0, num_epochs=100)
    model.load_state_dict(torch.load("model.pt")["model_state_dict"])
    task_to_programs = {task_name : [Program.parse(p) for p in program_strings] for task_name,program_strings in sample_decode(model, dataset, batch_size, n=10).items()}
    
    # run sampled programs with ocaml
    homeDirectory = "/".join(os.path.abspath(__file__).split("/")[:-4])
    dataDirectory = "arc_data/data/"
    tasks = retrieveARCJSONTasks(dataDirectory + 'training', useEvalExamplesForTraining=False, filenames=None)
    # getting actual Task objects instead of just task_name (string)
    train_tasks = [t for t in tasks if t.name in task_to_programs]
    task_to_log_likelihoods = execute_programs(train_tasks, grammar, task_to_programs)
    for t,log_likelihoods in task_to_log_likelihoods.items():
        print(t, log_likelihoods)
        print("----------------------------------------------------------")



