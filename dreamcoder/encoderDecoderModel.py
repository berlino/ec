import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

from dreamcoder.decoderUtils import getPrimitivesOfType
from dreamcoder.program import Program
from dreamcoder.recognition import RecognitionModel

MAX_PROGRAM_LENGTH = 30

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
        print("popping right to left")
        return self.stack.__str__()

    def __iter__(self):
        for x in self.stack:
            yield x

class ProgramDecoder(nn.Module):
    def __init__(self, embedding_size, grammar, max_program_length, encoderOutputSize):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.grammar = grammar
        self.max_program_length = max_program_length
        self.encoderOutputSize = encoderOutputSize

        # theoretically there could be infinitely nested lambda functions but we assume that
        # we can't have lambdas within lambdas
        self.primitiveToIdx = {
            "START": 0,
            "INPUT": 1,
            "LAMBDA": 2,
            "LAMBDA_INPUT": 3
        }
        offset = len(self.primitiveToIdx)
        for i,p in enumerate(grammar.primitives):
            self.primitiveToIdx[p] = i+offset

        self.IdxToPrimitive = {i:p for p,i in self.primitiveToIdx.items()}

        self.token_attention = nn.MultiheadAttention(self.embedding_size, 1)
        self.OutputTokenEmbeddings = nn.Embedding(len(self.primitiveToIdx), self.embedding_size)

        self.linearly_transform_query = nn.Linear(self.embedding_size + self.encoderOutputSize, self.embedding_size)
    
    def getKeysMask(self, nextTokenType, lambdaVarsTypeStack, lambdaVarInScope, request):
        """
        Given the the type of the next token we want and the stack of variables returns a mask (where we attend over 0s and not over -INF)
        which enforces the type constraints for the next token. 
        """

        possibleNextPrimitives = getPrimitivesOfType(nextTokenType, self.grammar)
        keys_mask = torch.full((1,len(self.primitiveToIdx)), -float("Inf"))
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

    def forward(self, encoderOutput, request, mode, targets):

        programTokenSeq = []
        totalScore = 0
        nextTokenTypeStack = Stack()
        lambdaVarsTypeStack = Stack()
        parentTokenStack = Stack()
        lambdaIndexInStack = float("inf")

        def forwardNextToken(encoderOutput, ppEncoding, nextTokenType, mode, targetToken):

            # print("encoderOutput shape: {}".format(encoderOutput.size()))
            # print("ppEncoding shape: {}".format(ppEncoding.size()))

            query = torch.cat((encoderOutput,ppEncoding), dim=1).unsqueeze(1)
            query = self.linearly_transform_query(query)

            # print("query shape: {}".format(query.size()))
            keys = self.OutputTokenEmbeddings.weight.unsqueeze(0)
            # print("keys shape: {}".format(keys.size()))

            # we only care about attnOutputWeights so values could be anything
            values = keys
            keys_mask = self.getKeysMask(nextTokenType, lambdaVarsTypeStack, lambdaIndexInStack <= len(parentTokenStack), request)
            # print('keys mask', keys_mask)
            _, attnOutputWeights = self.token_attention(query, keys, values, key_padding_mask=None, need_weights=True, attn_mask=keys_mask)
            # print("attention_output weights: {}".format(attnOutputWeights))

            # sample from attention weight distribution enforcing type constraints 
            nextTokenDist = Categorical(probs=attnOutputWeights[0, :])
            
            if mode == "sample":
                # get primitives of correct type
                nextTokenIdx = nextTokenDist.sample()
                return nextTokenIdx

            elif mode == "score":
                # notation of target program does not specify what type VAR is but we can infer from partial program and score the correct one
                if targetToken == "VAR":
                    targetTokenIdx = self.primitiveToIdx["INPUT"] if nextTokenType == request.functionArguments()[0] else self.primitiveToIdx["LAMBDA_INPUT"]
                elif targetToken == "LAMBDA":
                    targetTokenIdx = self.primitiveToIdx["LAMBDA"]
                else:
                    targetTokenIdx = self.primitiveToIdx[targetToken]
                targetTokenIdx = torch.LongTensor([targetTokenIdx])
                score = nextTokenDist.log_prob(targetTokenIdx)

                return targetTokenIdx.item(), score

        parentTokenIdx = self.primitiveToIdx["START"]
        parentTokenEmbedding = self.OutputTokenEmbeddings(torch.LongTensor([parentTokenIdx]))
        nextTokenTypeStack.push(request.returns())
        parentTokenStack.push(parentTokenIdx)

        while len(nextTokenTypeStack) > 0:

            # print("--------------------------------------------------------------------------------")            
            # print(" ".join([str(self.IdxToPrimitive[idx]) for idx in programTokenSeq]))
            # print("TypeStack: {}".format(nextTokenTypeStack))
            # print("parentTokenStack: {}".format([self.IdxToPrimitive[idx] for idx in parentTokenStack]))
            # print("lambdaIndexInStack: {}\n len(parentTokenStack): {}".format(lambdaIndexInStack, len(parentTokenStack)))

            # get type of next token
            nextTokenType = nextTokenTypeStack.pop()
            # get parent primitive of next token
            parentTokenIdx = parentTokenStack.pop()
            # sample next token
            parentTokenEmbedding = self.OutputTokenEmbeddings(torch.LongTensor([parentTokenIdx]))
            # TODO make compatible with batch
            nextTokenIdx, score = forwardNextToken(encoderOutput, parentTokenEmbedding, nextTokenType, mode, targets[0][len(programTokenSeq)])
            programTokenSeq.append(nextTokenIdx)
            totalScore += score

            nextToken = self.IdxToPrimitive[nextTokenIdx]
            if nextToken == "START":
                assert Exception("Should never sample START")
            elif nextToken == "INPUT":
                pass
            elif nextToken == "LAMBDA_INPUT":
                lambdaVarsTypeStack.pop()
            elif nextToken == "LAMBDA":
                # assume lambdas are used only with one argument
                # print("nextTokenType", nextTokenType)
                assert len(nextTokenType.functionArguments()) == 1
                lambdaVarsTypeStack.push(nextTokenType.functionArguments()[0])
                nextTokenTypeStack.push(nextTokenType.returns())
                # once the stack is this length again it means the lambda function has been synthesized LAMBDA_INPUT
                # can no longer be used
                lambdaIndexInStack = len(parentTokenStack)
                # technically the parent token is LAMBDA but that carries no information so we use the grandparent
                parentTokenStack.push(parentTokenIdx)

            elif nextToken.tp.isArrow() and not(nextTokenType.isArrow()):
                # print("sampled function when desired type is constant -> add function arguments to type stack")
                for arg in nextToken.tp.functionArguments()[::-1]:
                    nextTokenTypeStack.push(arg)
                    # keep track of parent function for which existing one is an argument
                    parentTokenStack.push(nextTokenIdx)
            else:
                pass

            # lambda function was synthesised so can no longer use LAMBDA_INPUT
            if len(parentTokenStack) <= lambdaIndexInStack:
                lambdaIndexInStack = float("inf")

            # program did not terminate within 20 tokens
            if len(programTokenSeq) > MAX_PROGRAM_LENGTH:
                print("---------------- Failed to find program < {} tokens ----------------------------".format(MAX_PROGRAM_LENGTH))
                return None

        # print('------------------------ Decoded below program ---------------------------------')
        # print(" ".join([str(self.IdxToPrimitive[idx]) for idx in programTokenSeq]))
        # print('--------------------------------------------------------------------------------')
        return programTokenSeq, totalScore

class Encoder(nn.Module):
    def __init__(self, featureExtractor, embedding_size=128):
        super().__init__()

        self.embedding_size = embedding_size
        self.featureExtractor = featureExtractor
        # Sanity check - make sure that all of the parameters of the
        # feature extractor were added to our parameters as well
        if hasattr(featureExtractor, 'parameters'):
            for parameter in featureExtractor.parameters():
                assert any(myParameter is parameter for myParameter in self.parameters())

    def forward(self, tasks):
        featureVectors = torch.stack(self.featureExtractor.featuresOfTasks(tasks))
        # print("encoded featureVectors size: {}".format(featureVectors.size()))
        return torch.Tensor(featureVectors)


class EncoderDecoderModel(nn.Module):
    """
    Program Synthesis model that takes in a feature vector (encoding) from which it outputs (decodes) a program. 
    By default inherits methods from RecognitionModel and overwrites as needed.

    Encoder:
        1. Encode IO examples with existing ARC CNN
        2. Encode NL description with existing T5 encoder

    Decoder:
        1. Use encoder output + partial program encoding as attention query, and NL description words / BERT encodings as keys and values
        2. Concatenate output from 1 with query from 1
        3. Use output from 2 as query and token embeddings as keys to get attention vector
    """

    def __init__(self, featureExtractor, grammar, cuda=False,
        previousRecognitionModel=None, id=0, embedding_size=128, program_size=128):
        super().__init__()
        
        self.id = id
        self.use_cuda = cuda

        self.encoder = Encoder(featureExtractor, embedding_size)
        # there are three additional tokens, one for the start token input grid (tgridin) variable and the other for input 
        # variable of lambda expression (assumes no nested lambdas)
        self.decoder = ProgramDecoder(embedding_size=embedding_size, grammar=grammar, max_program_length=100, 
            encoderOutputSize=featureExtractor.outputDimensionality)

        if cuda: self.cuda()

        if previousRecognitionModel:
            raise NotImplementedError()
            # self._MLP.load_state_dict(previousRecognitionModel._MLP.state_dict())
            # self.featureExtractor.load_state_dict(previousRecognitionModel.featureExtractor.state_dict())


    def forward(self, tasks, mode, targets=None):

        encoderOutputs = self.encoder(tasks)
        # assume all tasks are the same type
        assert all([t.request for t in tasks])
        # print("encoderOutputs shape: {}".format(encoderOutputs.size()))
        programTokenSequences = self.decoder(encoderOutputs, tasks[0].request, mode, targets)
        return programTokenSequences

    def train(self, tasks):

        for t in tasks:
            programTokenSeq = self(t)

