import pickle
import torch
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchviz import make_dot

from dreamcoder.domains.arc.arcPrimitives import basePrimitives, leafPrimitives, moreSpecificPrimitives, tgridin, tgridout
from dreamcoder.domains.arc.utilsPostProcessing import resume_from_path
from dreamcoder.grammar import Grammar
from dreamcoder.program import Program
from dreamcoder.recognition import RecognitionModel
from dreamcoder.type import arrow
from dreamcoder.utilities import ParseFailure

from larc.decoderUtils import get_primitives_of_type, program_to_token_sequence
from larc.encoder import LARCEncoder
from larc.larcDataset import *

MAX_PROGRAM_LENGTH = 30
MAX_NUM_IOS = 3

class Stack:
    """
    Stack data structure to enforce type constraints when decoding a program by sequentially sampling
    its primitives
    """

    def __init__(self):
        self.stack = []

    def pop(self):
        return self.stack.pop(-1)

    def push(self, tp):
        self.stack.append(tp)
        return

    def toPop(self):
        if len(self.stack) > 0:
            return self.stack[-1]
        return None

    def __contains__(self, x):
        return x in self.stack

    def __len__(self):
        return len(self.stack)

    def __repr__(self):
        return self.stack.__str__()

    def __iter__(self):
        for x in self.stack:
            yield x


class ProgramDecoder(nn.Module):
    def __init__(self, embedding_size, batch_size, grammar, request, cuda, device, max_program_length, encoderOutputSize, primitive_to_idx):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.grammar = grammar
        self.request = request
        self.device = device
        self.max_program_length = max_program_length
        self.encoderOutputSize = encoderOutputSize
        self.batch_size = batch_size

        # theoretically there could be infinitely nested lambda functions but we assume that
        # we can't have lambdas within lambdas
        self.primitiveToIdx = primitive_to_idx
        self.idxToPrimitive = {idx: primitive for primitive,idx in self.primitiveToIdx.items()}

        self.token_attention = nn.MultiheadAttention(self.embedding_size, 1)
        self.output_token_embeddings = nn.Embedding(len(self.primitiveToIdx), self.embedding_size)

        self.linearly_transform_query = nn.Linear(self.embedding_size + self.encoderOutputSize, self.embedding_size)

        if cuda: self.cuda()
        
    def getKeysMask(self, nextTokenType, lambdaVarsTypeStack, lambdaVarInScope, request):
        """
        Given the the type of the next token we want and the stack of variables returns a mask (where we attend over 0s and not over -INF)
        which enforces the type constraints for the next token. 
        """

        possibleNextPrimitives = get_primitives_of_type(nextTokenType, self.grammar)
        keys_mask = torch.full((1,len(self.primitiveToIdx)), -float("Inf"), device=self.device)
        keys_mask[:, [self.primitiveToIdx[p] for p in possibleNextPrimitives]] = 0

        # if the next token we need is a function then add creating a lambda abstraction as a possible action to sample.
        # TODO: change to non-hacky version where we always have the option to create lambda abstraction
        if nextTokenType.isArrow() and torch.all(keys_mask == -float("Inf")):
            keys_mask[:, self.primitiveToIdx["LAMBDA"]] = 0

        # if the type we need is the same as the input (tgridin) then add this as possible action to sample
        if nextTokenType == request.functionArguments()[0]:
            keys_mask[:, self.primitiveToIdx["INPUT"]] = 0

        # if the we need is the same as the lambda variable then add this as a possible action to sample.
        # the second clause ensures that LAMBDA_INPUT is in scope i.e. not outside lambda expression
        if nextTokenType == lambdaVarsTypeStack.toPop() and lambdaVarInScope:
            keys_mask[:, self.primitiveToIdx["LAMBDA_INPUT"]] = 0

        return keys_mask

    def forward(self, encoderOutput, mode, targets):

        programTokenSeq = [torch.tensor([self.primitiveToIdx["START"]], device=self.device)]
        programStringsSeq = ["lambda"]
        totalScore = 0.0
        nextTokenTypeStack = Stack()
        lambdaVarsTypeStack = Stack()
        
        parentTokenStack = Stack()

        lambdaIndexInStack = float("inf")
        
        openParenthesisStack = Stack()
        # openParenthesisStack.push(0)

        def forwardNextToken(encoderOutput, ppEncoding, nextTokenType, mode, targetTokenIdx):

            query = torch.cat((encoderOutput,ppEncoding)).reshape(1,1,-1)
            query = self.linearly_transform_query(query)

            # unsqueeze in 1th dimension corresponds to batch_size=1
            keys = self.output_token_embeddings.weight.unsqueeze(1)

            # we only care about attnOutputWeights so values could be anything
            values = keys
            keys_mask = self.getKeysMask(nextTokenType, lambdaVarsTypeStack, lambdaIndexInStack <= len(parentTokenStack), self.request)
            # print('keys mask shape: ', keys_mask.size())
            _, attnOutputWeights = self.token_attention(query, keys, values, key_padding_mask=None, need_weights=True, attn_mask=keys_mask)
            # print("attention_output weights: {}".format(attnOutputWeights))

            # sample from attention weight distribution enforcing type constraints 
            nextTokenDist = Categorical(probs=attnOutputWeights)
            
            if mode == "sample":
                # get primitives of correct type
                nextTokenIdx = nextTokenDist.sample()
                return nextTokenIdx, 0.0

            elif mode == "score":
                score = nextTokenDist.log_prob(targetTokenIdx)
                # print("targetTokenIdx: {} () -> {}".format(targetTokenIdx, self.idxToPrimitive[targetTokenIdx.item()], score))

                return targetTokenIdx, score

        parentTokenIdx = torch.tensor([self.primitiveToIdx["START"]], device=self.device)
        parentTokenEmbedding = self.output_token_embeddings(parentTokenIdx)[0, :]
        nextTokenTypeStack.push(self.request.returns())
        parentTokenStack.push(parentTokenIdx)

        while len(nextTokenTypeStack) > 0:

            # get type of next token
            nextTokenType = nextTokenTypeStack.pop()
            parentTokenIdx = parentTokenStack.pop()

            # sample next token
            partialProgramEmbedding = self.output_token_embeddings(torch.tensor([parentTokenIdx], device=self.device))

            if mode == "score":
                # TODO: Make compatible with batching
                targetTokenIdx = targets[len(programTokenSeq)]
            else:
                targetTokenIdx = None

            # TODO make compatible with batch
            nextTokenIdx, score = forwardNextToken(encoderOutput, parentTokenEmbedding, nextTokenType, mode, targetTokenIdx)
            totalScore += score
            
            programTokenSeq.append(nextTokenIdx)

            nextToken = self.idxToPrimitive[nextTokenIdx.item()]

            if nextToken == "START":
                assert Exception("Should never sample START")
            elif nextToken == "INPUT":
                programStringsSeq.append("${}".format(len(lambdaVarsTypeStack)))
                pass
            elif nextToken == "LAMBDA_INPUT":
                programStringsSeq.append("${}".format(len(lambdaVarsTypeStack) - 1))
                lambdaVarsTypeStack.pop()
            elif nextToken == "LAMBDA":
                # assume lambdas are used only with one argument
                assert len(nextTokenType.functionArguments()) == 1
                for arg in nextTokenType.functionArguments():
                    lambdaVarsTypeStack.push(arg)
                nextTokenTypeStack.push(nextTokenType.returns())
                # once the stack is this length again it means the lambda function has been synthesized LAMBDA_INPUT
                # can no longer be used
                lambdaIndexInStack = len(parentTokenStack)
                openParenthesisStack.push(len(parentTokenStack))
                # technically the parent token is LAMBDA but that carries no information so we use the grandparent
                parentTokenStack.push(torch.tensor([self.primitiveToIdx["LAMBDA"]], device=self.device))
                programStringsSeq.append("(lambda")
            else:
                sampledToken = Program.parse(nextToken)
                if sampledToken.tp.isArrow() and not(nextTokenType.isArrow()):
                    # print("sampled function when desired type is constant -> add function arguments to type stack")
                    programStringsSeq.append("(")
                    # keep track of how many open left parenthesis they are by storing length of the parent token stack when its created
                    openParenthesisStack.push(len(parentTokenStack))
                    for arg in sampledToken.tp.functionArguments()[::-1]:
                        nextTokenTypeStack.push(arg)
                        # keep track of parent function for which existing one is an argument
                        parentTokenStack.push(nextTokenIdx)
                programStringsSeq.append(str(sampledToken))

            # lambda function was synthesised so can no longer use LAMBDA_INPUT
            if len(parentTokenStack) <= lambdaIndexInStack:
                lambdaIndexInStack = float("inf")

            # the openParenthesisStack will always contain numbers in ascending order
            while len(openParenthesisStack) > 0 and len(parentTokenStack) <= openParenthesisStack.toPop():
                programStringsSeq.append(")")
                openParenthesisStack.pop()

            # print("--------------------------------------------------------------------------------")            
            # print("programTokenSeq", [str(self.idxToPrimitive[idx.item()]) for idx in programTokenSeq])
            # print("openParenthesisStack", openParenthesisStack)
            # print("programStringsSeq", programStringsSeq)
            # print("TypeStack: {}".format(nextTokenTypeStack))
            # print("parentTokenStack: {}".format([self.idxToPrimitive[idx.item()] for idx in parentTokenStack]))
            # print("lambdaIndexInStack: {}\n len(parentTokenStack): {}".format(lambdaIndexInStack, len(parentTokenStack)))

            # program did not terminate within 20 tokens
            if len(programTokenSeq) > MAX_PROGRAM_LENGTH:
                print("---------------- Failed to find program < {} tokens ----------------------------".format(MAX_PROGRAM_LENGTH))
                return None

        # print('------------------------ Decoded below program ---------------------------------')
        # print(" ".join([str(self.IdxToPrimitive[idx]) for idx in programTokenSeq]))
        # print('--------------------------------------------------------------------------------')
        return programTokenSeq, " ".join(programStringsSeq), totalScore

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
        self.decoder = ProgramDecoder(embedding_size=program_embedding_size, batch_size=batch_size, grammar=grammar, request=request, cuda=cuda, device=device, max_program_length=MAX_PROGRAM_LENGTH, 
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

def sample_decode(model, tasks, batch_size, n=1000):

    data_loader = DataLoader(tasks, batch_size=batch_size, drop_last=True)
    task_to_token_sequences = {}

    for batch in data_loader:

        encoderOutputs = model.encoder(batch["io_grids"], batch["test_in"], batch["desc_tokens"])

        # iterate through each task in the batch
        for i in range(batch_size):
            
            task = batch["name"][i]
            task_to_token_sequences[task] = []
            print("Decoding task {}".format(task))

            # sample n times for each task
            for k in range(n):
                output = model.decoder(encoderOutputs[:, i], "sample", None)
                # if the we reach the MAX_PROGRAM_LENGTH and the program tree still has holes continue
                if output is None:
                    continue
                else:
                    token_sequence = output[0]
                    program_string = output[1]
                    task_to_token_sequences[task].append((token_sequence, program_string))

    return task_to_token_sequences

# def beam_search_decode(model, tasks, ba)


    #     task_to_samples[task["name"][0]] = []
    #     token_sequences, scores = model(io_grids=batch["io_grids"], test_in=batch["test_in"], desc_tokens=batch["desc_tokens"], mode="sample", targets=batch['programs'])
    #     for i in range(n):
    #         res = model(io_grids=task["io_grids"], test_in=task["test_in"], desc_tokens=task["desc_tokens"], mode="sample", targets=task['programs'])
    #         if res is None:
    #             continue
    #         else:
    #             token_sequences, scores = res
    #             task_to_samples[task["name"][0]].append(res[0])
    #     print("Failed to sample syntactically valid program after {} tries for task {}".format(n, task["name"][0]))
    # print(task_to_samples)
    # return task_to_samples


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
    batch_size = 64

    if use_cuda: 
        assert torch.cuda.is_available()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # load grammar / DSL
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

    # load dataset
    tasks_dir = "data/larc/tasks_json"
    json_file_name = "data/arc/prior_enumeration_frontiers_8hr.json"
    task_to_programs_json = json.load(open(json_file_name, 'r'))
    task_to_programs = load_task_to_programs_from_frontiers_json(grammar, token_to_idx,
        max_program_length=MAX_PROGRAM_LENGTH, task_to_programs_json=task_to_programs_json)
    larc_train_dataset = LARC_Cell_Dataset(tasks_dir, tasks_subset=None, num_ios=MAX_NUM_IOS, resize=(30, 30), task_to_programs=task_to_programs, device=device)
    dataset = larc_train_dataset
 
    # model = train_imitiation_learning(model, dataset, batch_size=batch_size, lr=1e-3, weight_decay=0.0, num_epochs=100)

    model.load_state_dict(torch.load("model.pt")["model_state_dict"])
    task_to_samples = sample_decode(model, dataset, batch_size, n=10)
    
    for task_name, samples in task_to_samples.items():
        print("=============================== task {} ====================================".format(task_name))
        # print("Ground truth programs")
        # for program in task["programs"]:
        #     print([idx_to_token[idx.item()] for idx in program])

        print("\nSamples")
        # TODO: fix to work with batching
        for sample in samples:
            token_sequence, program_string = sample
            print(program_string)
            p = Program.parse("(" + program_string + ")")
            print(p)

        print("Ground truth programs")
        for program in task_to_programs_json[task_name]:
            print(program)


