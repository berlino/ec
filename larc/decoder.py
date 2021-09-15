from torch.distributions.categorical import Categorical
import torch.nn as nn
from torch.utils.data import DataLoader

from larc.decoderUtils import *
from larc.larcDataset import collate

MAX_PROGRAM_LENGTH = 30

class PartialProgram:
    def __init__(self, 
        primitiveToIdx,
        requestReturnType,
        device, 
        programTokenSeq=None, 
        programStringsSeq=None, 
        nextTokenTypeStack=None, 
        lambdaVarsTypeStack=None, 
        openParenthesisStack=None,
        parentTokenStack=None,
        totalScore=None):

        if programTokenSeq is None:
            self.programTokenSeq = [torch.tensor([primitiveToIdx["START"]], device=device)]
        if programStringsSeq is None: 
            self.programStringsSeq = ["(lambda"]
        if totalScore is None:
            self.totalScore = 0.0

        if nextTokenTypeStack is None:
            self.nextTokenTypeStack = Stack()
            self.nextTokenTypeStack.push(requestReturnType)
        if lambdaVarsTypeStack is None:
            self.lambdaVarsTypeStack = Stack()
        if openParenthesisStack is None:
            self.openParenthesisStack = Stack()
        if parentTokenStack is None:
            self.parentTokenStack = Stack()
            parentTokenIdx = torch.tensor([primitiveToIdx["START"]], device=device)
            self.parentTokenStack.push(parentTokenIdx)

    def processNextToken(self, nextToken, score, primitiveToIdx, device):

        if nextToken == "START":
            assert Exception("Should never sample START")
        elif nextToken == "INPUT":
            self.programStringsSeq.append("${}".format(len(self.lambdaVarsTypeStack)))
            pass
        elif nextToken == "LAMBDA_INPUT":
            # TODO: fix so that we don't deterministically choose the first one
            self.programStringsSeq.append("${}".format(self.lambdaVars[0]))
            # lambdaVarsTypeStack.pop()
        elif nextToken == "LAMBDA":
            # assume lambdas are used only with one argument
            assert len(self.nextTokenType.functionArguments()) == 1
            for arg in self.nextTokenType.functionArguments():
                self.lambdaVarsTypeStack.push(arg)
            self.nextTokenTypeStack.push(self.nextTokenType.returns())
            # boolean stores whether this parenthesis corresponds to a lambda which is needed to manage LAMBDA_INPUT score
            self.openParenthesisStack.push((len(self.parentTokenStack), True))
            # technically the parent token is LAMBDA but that carries no information so we use the grandparent
            self.parentTokenStack.push(torch.tensor([primitiveToIdx["LAMBDA"]], device=device))
            self.programStringsSeq.append("(lambda")
        else:
            sampledToken = Program.parse(nextToken)
            if sampledToken.tp.isArrow() and not(self.nextTokenType.isArrow()):
                # print("sampled function when desired type is constant -> add function arguments to type stack")
                self.programStringsSeq.append("(")
                # keep track of how many open left parenthesis they are by storing length of the parent token stack when its created
                # boolean stores whether this parenthesis corresponds to a lambda which is needed to manage LAMBDA_INPUT score
                self.openParenthesisStack.push((len(self.parentTokenStack), False))
                for arg in sampledToken.tp.functionArguments()[::-1]:
                    self.nextTokenTypeStack.push(arg)
                    # keep track of parent function for which existing one is an argument
                    self.parentTokenStack.push(primitiveToIdx[nextToken])
            self.programStringsSeq.append(str(sampledToken))

        self.programTokenSeq.append(primitiveToIdx[nextToken])

        # the openParenthesisStack will always contain numbers in ascending order
        while len(self.openParenthesisStack) > 0 and len(self.parentTokenStack) <= self.openParenthesisStack.toPop()[0]:
            self.programStringsSeq.append(")")
            _, isLambda = self.openParenthesisStack.pop()
            # if we are closing the scope of a lambda, we want to remove the LAMBDA_INPUT from the scope
            # poppedLambda ensures we only do this once
            if isLambda:
                self.lambdaVarsTypeStack.pop()

        self.totalScore += score

        return self

class Decoder(nn.Module):
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
        
    def getKeysMask(self, nextTokenType, lambdaVarsTypeStack, request):
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
        lambdaVars = []
        # by default this is from top of stack to bottom
        for i,lambdaVarType in enumerate(lambdaVarsTypeStack):
            if nextTokenType == lambdaVarType:
                keys_mask[:, self.primitiveToIdx["LAMBDA_INPUT"]] = 0
                lambdaVars.append(i)

        return keys_mask, lambdaVars

    # def forward(self, mode, targets, programTokenSeq, programStringsSeq, totalScore, nextTokenTypeStack, lambdaVarsTypeStack, parentTokenStack, openParenthesisStack, encoderOutput=None):

    def forward(self, encoderOutput, pp):

        # get type of next token
        nextTokenType = pp.nextTokenTypeStack.pop()

        # assumes batch size of 1
        parentTokenEmbedding = self.output_token_embeddings(torch.tensor([pp.parentTokenStack.pop()], device=self.device))[0, :]

        query = torch.cat((encoderOutput, parentTokenEmbedding)).reshape(1,1,-1)
        query = self.linearly_transform_query(query)

        # unsqueeze in 1th dimension corresponds to batch_size=1
        keys = self.output_token_embeddings.weight.unsqueeze(1)

        # we only care about attnOutputWeights so values could be anything
        values = keys
        keys_mask, lambdaVars = self.getKeysMask(nextTokenType, pp.lambdaVarsTypeStack, self.request)
        # print('keys mask shape: ', keys_mask.size())
        _, attnOutputWeights = self.token_attention(query, keys, values, key_padding_mask=None, need_weights=True, attn_mask=keys_mask)
        # print("attention_output weights: {}".format(attnOutputWeights))

        # update partial program information (now that we know what type next token is)
        pp.nextTokenType = nextTokenType
        pp.lambdaVars = lambdaVars

        return attnOutputWeights, pp


def sample_decode(decoder, encoderOutput):

    pp = PartialProgram(decoder.primitiveToIdx, decoder.request.returns(), decoder.device)

    while len(pp.nextTokenTypeStack) > 0:

        # sample next token
        attnOutputWeights, pp = decoder.forward(encoderOutput, pp)

        nextTokenDist = Categorical(probs=attnOutputWeights)
        nextTokenIdx = nextTokenDist.sample()
        score = nextTokenDist.log_prob(nextTokenIdx)

        nextToken = decoder.idxToPrimitive[nextTokenIdx.item()]
        # update stacks
        pp.processNextToken(nextToken, score, decoder.primitiveToIdx, decoder.device)

        # program did not terminate within the allowed number of tokens
        if len(pp.programTokenSeq) > MAX_PROGRAM_LENGTH:
            print("---------------- Failed to find program < {} tokens ----------------------------".format(MAX_PROGRAM_LENGTH))
            return None
    return pp


def sample(model, tasks, batch_size, n=1000):

    data_loader = DataLoader(tasks, batch_size=batch_size, collate_fn =lambda x: collate(x, False), drop_last=True)
    task_to_program_strings = {}

    for batch in data_loader:

        encoderOutputs = model.encoder(batch["io_grids"], batch["test_in"], batch["desc_tokens"])

        # iterate through each task in the batch
        for i in range(batch_size):
            
            task = batch["name"][i]
            task_to_program_strings[task] = []

            # sample n times for each task
            for k in range(n):
                output = sample_decode(model.decoder, encoderOutputs[:, i])
                # if the we reach the MAX_PROGRAM_LENGTH and the program tree still has holes continue
                if output is None:
                    continue
                else:
                    # TODO: fix; currently have hack to properly remove space where uncessary (python parser is more flexible than ocaml parser)
                    program_string = " ".join(output.programStringsSeq + [")"])
                    program_string = str(Program.parse(program_string))
                    task_to_program_strings[task].append(program_string)

    return task_to_program_strings


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
