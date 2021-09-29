import os
import pickle
import random
import time
import torch
from torchviz import make_dot

import torch

from dreamcoder.domains.arc.main import retrieveARCJSONTasks
from dreamcoder.domains.arc.arcPrimitives import basePrimitives, leafPrimitives, moreSpecificPrimitives, tgridin, tgridout
from dreamcoder.domains.arc.utilsPostProcessing import resume_from_path
from dreamcoder.grammar import Grammar
from dreamcoder.program import Program
from dreamcoder.recognition import RecognitionModel
from dreamcoder.type import arrow
from dreamcoder.utilities import ParseFailure, EarlyStopping, parallelMap

from larc.decode import multicore_decode
from larc.decoder import MAX_PROGRAM_LENGTH
from larc.decoderUtils import *
from larc.encoderDecoder import EncoderDecoder
from larc.larcDataset import *
from larc.train import *

def main(args):

    use_cuda = args.pop("use_cuda")
    batch_size = args.pop("batch_size")
    lr = args.pop("lr")
    weight_decay = args.pop("weight_decay")
    beta = args.pop("beta")
    epochs_per_replay = args.pop("epochs_per_replay")
    beam_width = args.pop("beam_width")
    epsilon = args.pop("epsilon")
    max_programs_per_task = args.pop("max_p_per_task") 
    num_cpus = args.pop("num_cpus")
    num_cycles = args.pop("num_cycles")
    restrict_types = args.pop("restrict_types")
    rnn_decode = args.pop("rnn_decode")
    verbose = args.pop("verbose")
    seed = args.pop("seed")
    jumpstart = args.pop("jumpstart")

    # tasks_subset = ["67a3c6ac.json", "aabf363d.json"]
    tasks_subset = None

    torch.manual_seed(seed)
    random.seed(seed)

    if use_cuda: 
        assert torch.cuda.is_available()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print("Using {}".format(device)) 
    # # load grammar / DSL
    primitives_to_use = "base"
    if primitives_to_use == "base":
        primitives = basePrimitives() + leafPrimitives()
    else:
        primitives = basePrimitives() + leafPrimitives() + moreSpecificPrimitives()
    grammar = Grammar.uniform(primitives)

    # create dict from tokens to corresponding indices (to be used as indices for nn.Embedding)
    token_to_idx = {"PAD":0, "START": 1, "LAMBDA": 2, "LAMBDA_INPUT":3, "INPUT": 4}
    num_special_tokens = len(token_to_idx)
    token_to_idx.update({str(token):i+num_special_tokens for i,token in enumerate(grammar.primitives)})
    idx_to_token = {idx: token for token,idx in token_to_idx.items()}
  
    # load model
    request = arrow(tgridin, tgridout)
    print("Startin to load model")
    model = EncoderDecoder(grammar=grammar, request=request, cuda=use_cuda, device=device, rnn_decode=rnn_decode, 
        program_embedding_size=128, primitive_to_idx=token_to_idx)
    print("Finished loading model")

    # load tasks to check sampled programs against
    dataDirectory = "arc_data/data/"
    tasks = retrieveARCJSONTasks(dataDirectory + 'training', useEvalExamplesForTraining=False, filenames=None)

    # load already discovered programs
    tasks_dir = "data/larc/tasks_json"
    json_file_name = "data/arc/prior_enumeration_frontiers_8hr.json"
    task_to_programs_json = json.load(open(json_file_name, 'r'))
    task_to_programs = load_task_to_programs_from_frontiers_json(grammar, token_to_idx, max_program_length=MAX_PROGRAM_LENGTH, 
        task_to_programs_json=task_to_programs_json, device=device)
    print("Loaded task to programs")

    # tasks_with_programs = [t for t,programs in task_to_programs.items() if len(programs) > 0]
    # train_task_names, test_task_names = next(getKfoldSplit(tasks_with_programs, 0.8, 5))

    # load dataset for torch model
    larc_train_dataset_cpu = LARC_Cell_Dataset(tasks_dir, tasks_subset=tasks_subset, num_ios=MAX_NUM_IOS, resize=(30, 30), for_synthesis=True, beta=beta, task_to_programs=None, device=torch.device("cpu"))
    print("Finished loading dataset ({} samples)".format(len(larc_train_dataset_cpu)))
    
    task_to_correct_programs = {}
    for iteration in range(num_cycles):
        for start_idx, end_idx in get_batch_start_end_idxs(len(larc_train_dataset_cpu), batch_size):
            larc_train_dataset_batch_cpu = larc_train_dataset_cpu[start_idx:end_idx]
        
            if iteration == 0 and jumpstart:
                tasks_subset = [] if tasks_subset is None else tasks_subset
                task_to_programs = {k:v for k,v in task_to_programs.items() if ((len(v) > 0) and (k in tasks_subset))}
                task_to_programs = {k:v[:(min(max_programs_per_task, len(v)))] for k,v in task_to_programs.items()}
                task_to_correct_programs = task_to_programs
                print("{} initial tasks to learn from".format(len(task_to_programs)))
                if verbose:
                    for task, programs in task_to_correct_programs.items():
                        print("\n\n{}: {}".format(task, "\n".join([" ".join([idx_to_token[i] for i in p[0]]) for p in programs]))) 

            if len(task_to_correct_programs) > 0:
                model = model.to(device=torch.device("cuda"))
                model = train_experience_replay(model, task_to_correct_programs, tasks_dir=tasks_dir, beta=beta,
                   num_epochs=epochs_per_replay, lr=lr, weight_decay=weight_decay, batch_size=batch_size, device=device)
            
               torch.save({
                   'model_state_dict': model.state_dict(),
               }, 'data/larc/model_{}.pt'.format(iteration))
       
  
            # decode with randomized beam search
            print("Starting to decode")
            decode_start_time = time.time()
            model = model.to(device=torch.device("cpu"))
            task_to_decoded_programs, task_to_lls = multicore_decode(model, grammar, larc_train_dataset_batch_cpu, tasks, restrict_types=restrict_types, rnn_decode=rnn_decode, how="randomized_beam_search", 
                beam_width=beam_width, epsilon=epsilon, num_cpus=num_cpus, verbose=verbose)
            print("\nFinished Decoding in {}s \n".format(time.time() - decode_start_time))

            # experience replay train with discovered program
            task_to_correct_programs_iter = {}
            for task,log_likelihoods in task_to_lls.items():
                for i,ll in enumerate(log_likelihoods):
                    if ll == 0.0:
                        res = task_to_correct_programs_iter.get(task, [])
                        programName, program = task_to_decoded_programs[task][i]
                        print("Task {}: {}".format(task, programName))
 
                        paddedProgramTokenSeq = pad_token_seq(program.programTokenSeq, token_to_idx["PAD"], MAX_PROGRAM_LENGTH)
                        if use_cuda:
                            # put tensor on gpu for training
                            programScore = program.totalScore[0].to(device=torch.device("cuda"))
                        res.append((paddedProgramTokenSeq, programScore))

                        # add to library of discovered programs if it is not already there
                        task_programs = task_to_correct_programs.get(task, [])
                        existingTokenSequences = [tokenSeq for tokenSeq, score in task_programs]
                        if paddedProgramTokenSeq not in existingTokenSequences:
                            task_programs.append((paddedProgramTokenSeq, programScore))
                            task_to_correct_programs[task] = task_programs
                        task_to_correct_programs_iter[task] = res

            print("Decoded correct programs for {} tasks at iteration {}".format(len(task_to_correct_programs_iter), iteration))
            print("Decoded correct programs for {} tasks total".format(len(task_to_correct_programs)))

            print("task_to_correct_programs_iter", task_to_correct_programs_iter)
            print("task_to_correct_programs", task_to_correct_programs)

