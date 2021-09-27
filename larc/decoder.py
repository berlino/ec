from torch.distributions.categorical import Categorical
import torch.nn as nn
from torch.utils.data import DataLoader

from dreamcoder.utilities import parallelMap

from larc.beamSearch import randomized_beam_search_decode
from larc.decoderUtils import *
from larc.larcDataset import collate

MAX_PROGRAM_LENGTH = 30


class Decoder(nn.Module):
    def __init__(self, embedding_size, grammar, request, cuda, device, max_program_length, encoderOutputSize, primitive_to_idx):
        super(Decoder, self).__init__()
        
        self.embedding_size = embedding_size
        self.grammar = grammar
        self.request = request
        self.device = device
        self.max_program_length = max_program_length
        self.encoderOutputSize = encoderOutputSize

        # theoretically there could be infinitely nested lambda functions but we assume that
        # we can't have lambdas within lambdas
        self.primitiveToIdx = primitive_to_idx
        self.idxToPrimitive = {idx: primitive for primitive,idx in self.primitiveToIdx.items()}
        self.token_pad_value = len(self.idxToPrimitive)

        self.token_attention = nn.MultiheadAttention(self.embedding_size, 1, batch_first=False)
        self.output_token_embeddings = nn.Embedding(len(self.primitiveToIdx), self.embedding_size)

        self.linearly_transform_query = nn.Linear(self.embedding_size + self.encoderOutputSize, self.embedding_size)
        
        if cuda: self.cuda()


    def getKeysMask(self, nextTokenType, lambdaVarsTypeStack, request, device):
        """
        Given the the type of the next token we want and the stack of variables returns a mask (where we attend over 0s and not over -INF)
        which enforces the type constraints for the next token. 
        """

        possibleNextPrimitives = get_primitives_of_type(nextTokenType, self.grammar)
        keys_mask = torch.full((1,len(self.primitiveToIdx)), -float("Inf"), device=device)
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
        lambdaVars = []
        # by default this is from top of stack to bottom
        for i,lambdaVarType in enumerate(lambdaVarsTypeStack):
            if nextTokenType == lambdaVarType:
                keys_mask[:, self.primitiveToIdx["LAMBDA_INPUT"]] = 0
                lambdaVars.append(i)

        return keys_mask, lambdaVars


    def forward(self, encoderOutput, pp, parentTokenIdx, device, restrictTypes=True):
        

        if restrictTypes:
            # assumes batch size of 1
            parentTokenEmbedding = self.output_token_embeddings(torch.tensor([parentTokenIdx], device=device))[0, :]
            query = torch.cat((encoderOutput, parentTokenEmbedding), 0)
            # seq_length x batch_size x embed_dim
            query = self.linearly_transform_query(query).reshape(1,1,-1)

            # unsqueeze in 1th dimension corresponds to batch_size=1
            keys = self.output_token_embeddings.weight.unsqueeze(1)

            # we only care about attnOutputWeights so values could be anything
            values = keys
            
            # get type of next token
            nextTokenType = pp.nextTokenTypeStack.pop()
            keys_mask, lambdaVars = self.getKeysMask(nextTokenType, pp.lambdaVarsTypeStack, self.request, device)
            # print('keys mask shape: ', keys_mask.size())
            _, attnOutputWeights = self.token_attention(query, keys, values, key_padding_mask=None, need_weights=True, attn_mask=keys_mask)
            # print("attention_output weights: {}".format(attnOutputWeights))

            return attnOutputWeights, nextTokenType, lambdaVars, pp

        else:
            parentTokenEmbedding = self.output_token_embeddings(parentTokenIdx)
            
            query = torch.cat((encoderOutput, parentTokenEmbedding), 1)
            # seq_length x batch_size x embed_dim
            query = self.linearly_transform_query(query).unsqueeze(0)
            # seq_length x batch_size x embed dim (where keys are the same for all elements in batch
            keys = self.output_token_embeddings.weight.unsqueeze(1).expand(-1, encoderOutput.size(0), -1)
            values = keys

            _, attnOutputWeights = self.token_attention(query, keys, values, key_padding_mask=None, need_weights=True, attn_mask=None)

            return attnOutputWeights.squeeze()
