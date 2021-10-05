import math
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dreamcoder.program import Program
from dreamcoder.utilities import parallelMap
from neural_seq.beamSearch import randomized_beam_search_decode
from neural_seq.decoder import MAX_PROGRAM_LENGTH
from neural_seq.decoderUtils import execute_programs
from neural_seq.larcDataset import collate, get_batch_start_end_idxs

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

def score_decode(decoder, encoderOutput, targetTokens, rnn_decode, device):
    """
    Args:
        decoder (torch.nn.Module): torch Decoder
        encoderOutput (torch.tensor): batch_size x encoder_embed_dim
        targetTokens (torch.tensor): batch_size x MAX_PROGRAM_SEQ_LENGTH
        rnn_decode (bool): Whether to use RNN for decoding
    """
    batch_size = encoderOutput.size(0)
    scores = torch.empty(targetTokens.size(), device=device)

    if rnn_decode:
        # 1 x batch_size x embed_dim
        hidden = encoderOutput.unsqueeze(0)

    for i in range(MAX_PROGRAM_LENGTH):

        if rnn_decode:
            # batch_size x num_tokens, batch_size x hidden_embed_dim
            probs, hidden = decoder.forward_rnn(encoderOutput, pp=None, parentTokenIdx=targetTokens[:, i], 
            last_hidden=hidden, restrictTypes=False, device=device)
        else:
            # batch_size x num_tokens
            probs = decoder.forward(encoderOutput, pp=None, parentTokenIdx=targetTokens[:, i], restrictTypes=False, device=device)
    
        nextTokenDist = Categorical(probs=probs)
        scores[:, i] = -nextTokenDist.log_prob(targetTokens[:, i])

    # we don't want to take gradient steps on pad token after the program has already been sampled
    scorePerTaskInBatch = torch.empty(batch_size, device=device)
    for j in range(batch_size):
        paddingMask = targetTokens[j, :] != decoder.primitiveToIdx["PAD"]
        programTokenScores = scores[j, :][paddingMask]
        sampleScore = torch.sum(programTokenScores, axis=0)
        scorePerTaskInBatch[j] = sampleScore
    return scorePerTaskInBatch

def multicore_decode(model, grammar, dataset, tasks, restrict_types, rnn_decode, num_iter_beam_search=1, how="sample", n=10, beam_width=10, epsilon=0.1, num_cpus=1, verbose=False):
    
    def decode_helper(idx_pair, num_cpus, model):
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

            start_idx, end_idx = idx_pair
            data_loader = DataLoader(dataset[start_idx:end_idx], batch_size=end_idx-start_idx, collate_fn =lambda x: collate(x, False), drop_last=False, shuffle=True)

            task_to_programs = {}

            for batch in data_loader:
                
                # batch_size (1) x embed_dim
                encoderOutputs = model.encoder(batch["io_grids"], batch["test_in"], batch["desc_tokens"])
                
                # iterate through each task in the batch
                for i in range(encoderOutputs.size(0)):
   
                    task = batch["name"][i]
                    task_to_programs[task] = []

                    # run beam search num_iter_beam_search times
                    for j in range(num_iter_beam_search):

                        if how == "sample":
                            raise Exception("Not imlemented yet")
    # 
                        elif how == "randomized_beam_search":
                            beam_search_result = randomized_beam_search_decode(model.decoder, encoderOutputs[i:i+1, :], restrict_types=restrict_types, rnn_decode=rnn_decode, 
                                beam_width=beam_width, epsilon=epsilon, device=torch.device("cpu"))
                            if len(beam_search_result) == 0:
                                continue
                            else:
                                for (score, node) in beam_search_result:
                                    program_string = " ".join(node.programStringsSeq + [")"])
                                    program_string = str(Program.parse(program_string))
                                    task_to_programs[task].append((program_string, node))

            # if verbose and len(task_to_programs) > 0:
                # print("\nNumber of programs decoded per task")
                # print({t:len(programs) for t,programs in task_to_programs.items()})

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
        torch.multiprocessing.set_sharing_strategy('file_system')

    # calculate how many tasks to assign to each core    
    num_tasks_per_core = int(math.ceil(len(dataset) / num_cpus))
    idx_pairs = list(get_batch_start_end_idxs(len(dataset), num_tasks_per_core))
    
    # sharing model across cores so that it's not copied to each of them
    model.share_memory()
    parallel_results = parallelMap(
    num_cpus, 
    lambda idx_pair: decode_helper(idx_pair=idx_pair, num_cpus=num_cpus, model=model),
    idx_pairs,
    maxtasksperchild=True,
    memorySensitive=False)
    
    all_task_to_program, all_task_to_ll = {}, {}
    for task_to_ll, task_to_program in parallel_results:
        all_task_to_ll.update(task_to_ll)
        all_task_to_program.update(task_to_program)
   
    if verbose:
        if len(all_task_to_program) > 0:
            print("\nNumber of programs decoded per task")
            print({t:len(programs) for t,programs in all_task_to_program.items()})
        else:
            print("No programs discovered") 
    
    return all_task_to_program, all_task_to_ll
