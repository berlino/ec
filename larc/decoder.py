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

# TODO: Delete or fix bugs
def decode_single(decoder, encoderOutput, targetTokens=None):

    pp = PartialProgram(decoder.primitiveToIdx, decoder.request.returns(), decoder.device)

    while len(pp.nextTokenTypeStack) > 0:

        # sample next token
        attnOutputWeights, nextTokenType, lambdaVars, pp = decoder.forward(encoderOutput, pp)

        nextTokenDist = Categorical(probs=attnOutputWeights)

        score = -nextTokenDist.log_prob(nextTokenIdx)
        nextToken = decoder.idxToPrimitive[nextTokenIdx.item()]
        # update stacks
        pp.processNextToken(nextToken, nextTokenType, score, lambdaVars, decoder.primitiveToIdx, decoder.device)

        # program did not terminate within the allowed number of tokens
        if len(pp.programTokenSeq) > MAX_PROGRAM_LENGTH:
            print("---------------- Failed to find program < {} tokens ----------------------------".format(MAX_PROGRAM_LENGTH))
            return None
    return pp

def decode_score(decoder, encoderOutput, targetTokens, device):
    """
    Args:
        decoder (torch.nn.Module): torch Decoder
        encoderOutput (torch.tensor): batch_size x encoder_embed_dim
        targetTokens (torch.tensor): batch_size x MAX_PROGRAM_SEQ_LENGTH
    """
    batch_size = encoderOutput.size(0)

    scores = torch.empty(targetTokens.size(), device=device)

    for i in range(MAX_PROGRAM_LENGTH):
        # batch_size x num_tokens
        attnOutputWeights = decoder.forward(encoderOutput, pp=None, parentTokenIdx=targetTokens[:, i], restrictTypes=False, device=device)
        nextTokenDist = Categorical(probs=attnOutputWeights)
        scores[:, i] = -nextTokenDist.log_prob(targetTokens[:, i])

    # we don't want to take gradient steps on pad token after the program has already been sampled
    scorePerTaskInBatch = torch.empty(batch_size, device=device)
    for j in range(batch_size):
        paddingMask = targetTokens[j, :] != decoder.token_pad_value
        sampleScore = torch.sum(scores[j, :][paddingMask], axis=0)
        scorePerTaskInBatch[j] = sampleScore
    return scorePerTaskInBatch

def multicore_decode(model, grammar, dataset, tasks, batch_size, how="sample", n=10, beam_width=10, epsilon=0.1, num_cpus=1):
    
    def decode_helper(core_idx, num_cpus, model):
        """
        Returns:
             task_to_ll (dict): dictionary with entries (task_name, list of log_likelihoods) e.g. ("3459335.json", [0.0, -float("inf")])
             task_to_programs (dict): dictionary entries (task_name, list of program tuples) e.g. ("3459335.json", [("to_min_grid (...))", PartialProgram), ("to_original_grid_overlay(...)", PartialProgram)]
             
        """
        if num_cpus > 1:
            # required for torch.multiprocessing to work properly
            torch.set_num_threads(1)

        model.eval()
        with torch.no_grad():

            offset = len(dataset) // num_cpus
            data_loader = DataLoader(dataset[core_idx:core_idx+offset], batch_size=batch_size, collate_fn =lambda x: collate(x, False), drop_last=True, shuffle=True)
            print("loaded data: (core {})".format(core_idx))

            task_to_programs = {}

            for batch in data_loader:
                
                # batch_size x embed_dim
                encoderOutputs = model.encoder(batch["io_grids"], batch["test_in"], batch["desc_tokens"])
                print("got encoderOutputs (core idx {})".format(core_idx))
                # iterate through each task in the batch
                for i in range(batch_size):
                
                    task = batch["name"][i]
                    task_to_programs[task] = []

                    print("Decoding task {} (core_idx {})".format(task, core_idx))

                    if how == "sample":

                        # sample n times for each task
                        for k in range(n):
                            output = sample_decode(model.decoder, encoderOutputs[i, :])
                            # if the we reach the MAX_PROGRAM_LENGTH and the program tree still has holes continue
                            if output is None:
                                continue
                            else:
                                # TODO: fix; currently have hack to properly remove space where uncessary (python parser is more flexible than ocaml parser)
                                program_string = " ".join(output.programStringsSeq + [")"])
                                program_string = str(Program.parse(program_string))
                                task_to_programs[task].append((program_string, output))
    # 
                    elif how == "randomized_beam_search":
                        beam_search_result = randomized_beam_search_decode(model.decoder, encoderOutputs[i, :], beam_width=beam_width, epsilon=epsilon, num_end_nodes=n, device=torch.device("cpu"))
                        if len(beam_search_result) == 0:
                            continue
                        else:
                            for (score, node) in beam_search_result:
                                program_string = " ".join(node.programStringsSeq + [")"])
                                program_string = str(Program.parse(program_string))
                                task_to_programs[task].append((program_string, node))

            # run sampled programs with ocaml
            train_tasks = [t for t in tasks if t.name in task_to_programs]
            task_to_log_likelihoods = execute_programs(train_tasks, grammar, task_to_programs)
            task_to_ll = {}
            for item in task_to_log_likelihoods:
                # if item["task"] not in task_to_ll:
                #    task_to_ll[item["task"]] = []     
                # task_to_ll[item["task"]].append(item["log_likelihoods"])
                task_to_ll[item["task"]] = item["log_likelihoods"]    
            return task_to_ll, task_to_programs

    if num_cpus > 1:
        # required for torch.multiprocessing to work properly
        torch.set_num_threads(1)

    parallel_results = parallelMap(
    num_cpus, 
    lambda i: decode_helper(core_idx=i, num_cpus=num_cpus, model=model),
    range(num_cpus))
    
    all_task_to_program, all_task_to_ll = {}, {}
    for task_to_ll, task_to_program in parallel_results:
        all_task_to_ll.update(task_to_ll)
        all_task_to_program.update(task_to_program)
    
    for task, programs in all_task_to_program.items():
        print("\n{}: {} syntactically valid programs".format(task, len(programs)))
        for p in programs:
            print(p[0])
    
    return all_task_to_program, all_task_to_ll


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
