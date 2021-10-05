from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dreamcoder.utilities import parallelMap

from neural_seq.beamSearch import randomized_beam_search_decode
from neural_seq.decoderUtils import *
from neural_seq.larcDataset import collate

MAX_PROGRAM_LENGTH = 30

class Decoder(nn.Module):
    def __init__(self, embedding_size, grammar, request, cuda, device, max_program_length, encoderOutputSize,
        primitive_to_idx, use_rnn, hidden_size=64, num_layers=1, dropout=0.0):
        super(Decoder, self).__init__()
        
        self.embedding_size = embedding_size
        self.grammar = grammar
        self.request = request
        self.device = device
        self.max_program_length = max_program_length
        self.encoderOutputSize = encoderOutputSize
        self.num_layers = num_layers

        # theoretically there could be infinitely nested lambda functions but we assume that
        # we can't have lambdas within lambdas
        self.primitiveToIdx = primitive_to_idx
        self.idxToPrimitive = {idx: primitive for primitive,idx in self.primitiveToIdx.items()}
        self.output_token_embeddings = nn.Embedding(len(self.primitiveToIdx), self.embedding_size)

        if use_rnn:
            self.bridge = nn.Sequential(
                nn.Linear(self.encoderOutputSize, hidden_size),
                nn.Tanh()
            )

            self.gru = nn.GRU(input_size=self.embedding_size + self.encoderOutputSize, hidden_size=hidden_size,
                              num_layers=num_layers, dropout=dropout, batch_first=False, bias=True, bidirectional=False)
            self.fc = nn.Linear(hidden_size + encoderOutputSize,  hidden_size + encoderOutputSize)
            self.relu = nn.Tanh()
            self.out = nn.Linear(hidden_size + encoderOutputSize, len(self.primitiveToIdx))
        else:
            self.linearly_transform_query = nn.Linear(self.embedding_size + self.encoderOutputSize, self.embedding_size)
            self.token_attention = nn.MultiheadAttention(self.embedding_size, 1, batch_first=False)
        
        if cuda: self.cuda()


    def getKeysMask(self, nextTokenType, lambdaVarsTypeStack, request, device, bool_mask=False):
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

        if bool_mask:
            return keys_mask == -float("Inf"), lambdaVars
        else:
            return keys_mask, lambdaVars

    def forward_rnn(self, encoderOutput, pp, parentTokenIdx, last_hidden, device, restrictTypes=True):

        parentTokenEmbedding = self.output_token_embeddings(parentTokenIdx)
        # 1 x batch_size x embed_dim
        rnn_input = torch.cat((encoderOutput, parentTokenEmbedding), 1).unsqueeze(0)
        # seq_length x batch_size x embed_dim, seq_length x batch_size x embed_dim
        output, hidden = self.gru(rnn_input.contiguous(), last_hidden.contiguous())
        # batch_size x embed_dim
        output = output.squeeze(0)
        x = self.relu(self.fc(torch.cat([output, encoderOutput], 1)))
        logits = self.out(x)

        if restrictTypes:
            nextTokenType = pp.nextTokenTypeStack.pop()
            keys_mask, lambdaVars = self.getKeysMask(nextTokenType, pp.lambdaVarsTypeStack, self.request, device, bool_mask=True)

            for i in range(logits.size(0)):
                logits[i, :][keys_mask[i, :]] = -float("inf")
                
            probs = F.softmax(logits, dim=1)
            temp = probs.squeeze()
            
            # used for debugging
            # for i,token in sorted(list(self.idxToPrimitive.items()), key=lambda x: temp[x[0]]):
            #    if temp[i] > 0.0:
            #       print("{}: {}".format(token, temp[i]))
            return probs, hidden, nextTokenType, lambdaVars, pp

        else:
            probs = F.softmax(logits, dim=1)
            return probs, hidden


    def forward(self, encoderOutput, pp, parentTokenIdx, device, restrictTypes=True):

        parentTokenEmbedding = self.output_token_embeddings(parentTokenIdx)
        
        query = torch.cat((encoderOutput, parentTokenEmbedding), 1)
        # seq_length x batch_size x embed_dim
        query = self.linearly_transform_query(query).unsqueeze(0)
        # seq_length x batch_size x embed dim (where keys are the same for all elements in batch
        keys = self.output_token_embeddings.weight.unsqueeze(1).expand(-1, encoderOutput.size(0), -1)
        values = keys

        if restrictTypes:
            nextTokenType = pp.nextTokenTypeStack.pop()
            keys_mask, lambdaVars = self.getKeysMask(nextTokenType, pp.lambdaVarsTypeStack, self.request, device)
            _, attnOutputWeights = self.token_attention(query, keys, values, key_padding_mask=None, need_weights=True, attn_mask=keys_mask)
            return attnOutputWeights, nextTokenType, lambdaVars, pp
        else:
            _, attnOutputWeights = self.token_attention(query, keys, values, key_padding_mask=None, need_weights=True, attn_mask=None)
            return attnOutputWeights.squeeze()
