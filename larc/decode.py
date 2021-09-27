import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dreamcoder.program import Program
from dreamcoder.utilities import parallelMap
from larc.beamSearch import randomized_beam_search_decode
from larc.decoder import MAX_PROGRAM_LENGTH
from larc.decoderUtils import execute_programs
from larc.larcDataset import collate

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

def score_decode(decoder, encoderOutput, targetTokens, device):
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
                        raise Exception("Not imlemented yet")
    # 
                    elif how == "randomized_beam_search":
                        beam_search_result = randomized_beam_search_decode(model.decoder, encoderOutputs[i, :], beam_width=beam_width, 
                            epsilon=epsilon, device=torch.device("cpu"))
                        if len(beam_search_result) == 0:
                            continue
                        else:
                            for (score, node) in beam_search_result:
                                program_string = " ".join(node.programStringsSeq + [")"])
                                program_string = str(Program.parse(program_string))
                                print("program: ", program_string)
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