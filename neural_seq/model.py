import datetime
import dill
import json
import os
import pickle
import random
import time
import torch
from torchviz import make_dot

import torch

from dreamcoder.domains.arc.main import retrieveARCJSONTasks
from dreamcoder.domains.arc.arcPrimitives import basePrimitives, leafPrimitives, moreSpecificPrimitives, tgridin, tgridout
from dreamcoder.grammar import Grammar
from dreamcoder.program import Program
from dreamcoder.recognition import RecognitionModel
from dreamcoder.type import arrow
from dreamcoder.utilities import ParseFailure, EarlyStopping, parallelMap

from neural_seq.decode import multicore_decode
from neural_seq.decoder import MAX_PROGRAM_LENGTH
from neural_seq.decoderUtils import *
from neural_seq.encoderDecoder import EncoderDecoder
from neural_seq.larcDataset import *
from neural_seq.train import *

TRAIN_BATCH_SIZE = 8
NUM_EPOCHS_START = 50
TOP_N = 3

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
    num_iter_beam_search = args.pop("num_iter_beam_search")
    preload_frontiers = args.pop("preload_frontiers")
    limit_overfit = args.pop("limit_overfit")
    no_nl = args.pop("no_nl")
    no_io = args.pop("no_io")
    num_epochs_start = args.pop("num_epochs_start")
    resume = args.pop("resume")
    resume_iter = args.pop("resume_iter")
    fixed_epoch_pretrain = args.pop("fixed_epoch_pretrain")
    test = args.pop("test")
    test_decode_time = args.pop("test_decode_time")

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
    model = EncoderDecoder(grammar=grammar, request=request, cuda=use_cuda, device=device, rnn_decode=rnn_decode, use_nl=not no_nl, use_io=not no_io,
        program_embedding_size=128, primitive_to_idx=token_to_idx)
    print("Finished loading model")

    # load weights from file if resuming
    if resume is not None:
        outputDirectory = resume
        path = "{}/model_{}.pt".format(outputDirectory, resume_iter)
        state_dict = torch.load(path)["model_state_dict"]
        model.load_state_dict(state_dict)
        print("Resumed model at iteration {}, from {}".format(resume_iter, path))

    # load tasks
    data_directory = "arc_data/data/"
    larc_directory = "data/larc/"
    train_test_split_filename = "train_test_split.json"
    tasks = retrieveARCJSONTasks(data_directory + 'training', useEvalExamplesForTraining=True, filenames=None)
    tasks_no_eval_ex = retrieveARCJSONTasks(data_directory + 'training', useEvalExamplesForTraining=False, filenames=None)

    # load train and test task names
    train_test_split_dict = json.load(open(larc_directory + train_test_split_filename, "r"))
    train_task_names = [t for t in train_test_split_dict["train"]]
    test_task_names = [t for t in train_test_split_dict["test"]]
    assert len(set(train_task_names).intersection(set(test_task_names))) == 0

    pretraining = False
    # load already discovered programs
    if preload_frontiers is not None:
        task_to_programs = preload_frontiers_to_task_to_programs(preload_frontiers_filename=preload_frontiers)
        task_to_programs = process_task_to_programs(grammar, token_to_idx, max_program_length=MAX_PROGRAM_LENGTH, 
            task_to_programs=task_to_programs, device=device)
        print("Loaded task to programs")
        print(task_to_programs)
        pretraining = True

    # load dataset for torch model
    # load task_to_sentences file to use instead of default NL data (so as to use the same NL data as bigram model)
    task_to_sentences = json.load(open(larc_directory + "language.json", "r"))
    print("task_to_sentences length", len(task_to_sentences))

    # load train and test task names
    train_test_split_dict = json.load(open(larc_directory + train_test_split_filename, "r"))
    train_task_names = [t for t in train_test_split_dict["train"]]
    # we only want to test on tasks that NL annotations so that we can compare synthesizers that use different natural programs (e.g. IO vs IO + NL vs NL)
    test_task_names = [t for t in train_test_split_dict["test"] if t in task_to_sentences]
    assert len(set(train_task_names).intersection(set(test_task_names))) == 0

    tasks_subset = tasks_subset if tasks_subset is not None else train_task_names
    tasks_dir = "data/larc/tasks_json"

    if preload_frontiers is not None:
        pretrain_dataset_cpu = LARC_Cell_Dataset(tasks_dir, tasks_subset=list(task_to_programs.keys()), num_ios=MAX_NUM_IOS, resize=(30, 30), for_synthesis=True, beta=beta,
            task_to_programs=task_to_programs, device=torch.device("cpu"), task_to_sentences=task_to_sentences)
        pretrain_dataset_gpu = LARC_Cell_Dataset(tasks_dir, tasks_subset=list(task_to_programs.keys()), num_ios=MAX_NUM_IOS, resize=(30, 30), for_synthesis=True, beta=beta,
            task_to_programs=task_to_programs, device=torch.device("cuda"), task_to_sentences=task_to_sentences)
        print("Finished loading pre dataset ({} samples)".format(len(pretrain_dataset_gpu)))

    larc_train_dataset_cpu = LARC_Cell_Dataset(tasks_dir, tasks_subset=tasks_subset, num_ios=MAX_NUM_IOS, resize=(30, 30), for_synthesis=True, beta=beta, 
        task_to_programs=None, device=torch.device("cpu"), task_to_sentences=task_to_sentences)
    larc_test_dataset_cpu = LARC_Cell_Dataset(tasks_dir, tasks_subset=test_task_names, num_ios=MAX_NUM_IOS, resize=(30, 30), for_synthesis=True, beta=beta,
        task_to_programs=None, device=torch.device("cpu"), task_to_sentences=task_to_sentences)
    print("Finished loading distant supervision train dataset ({} samples)".format(len(larc_train_dataset_cpu)))
    print("Finished loading test dataset ({} samples)".format(len(larc_test_dataset_cpu)))


    if resume:
        outputDirectory = resume
    else:
        # make results directory
        timestamp = datetime.datetime.now().isoformat()
        outputDirectory = "experimentOutputs/larc_neural/%s" % timestamp
        os.system("mkdir -p %s" % outputDirectory)
    
    task_to_recently_decoded = {t:False for t in train_task_names}
    task_to_correct_programs_iter = {}
    task_to_correct_programs = dill.load(open("{}/task_to_correct_programs.pkl".format(outputDirectory), "rb")) if resume else {}

    if test:
        start_time = time.time()
        # key is task name, value is (program_name, PartialProgram)
        test_tasks_to_discovered_programs = {}
        for start_idx, end_idx in get_batch_start_end_idxs(len(larc_test_dataset_cpu), batch_size):
            larc_test_dataset_batch_cpu = larc_test_dataset_cpu[start_idx:end_idx]

            # decode with randomized beam search
            print("Starting to decode ({} - {})".format(start_idx, end_idx))
            decode_start_time = time.time()
            model = model.to(device=torch.device("cpu"))
            task_to_decoded_programs, task_to_lls = multicore_decode(model, grammar, larc_test_dataset_batch_cpu, tasks_no_eval_ex, restrict_types=restrict_types, rnn_decode=rnn_decode, num_iter_beam_search=num_iter_beam_search,
            how="randomized_beam_search", beam_width=beam_width, epsilon=0.0, num_cpus=num_cpus, verbose=verbose)
            print("\nFinished Decoding in {}s \n".format(time.time() - decode_start_time))
 
            # save discovered programs
            for t in task_to_decoded_programs.keys():
                for p_idx,ll in enumerate(task_to_lls[t]):
                    if ll == 0.0:
                        print("Found program: {} for task: {}".format(task_to_decoded_programs[t][p_idx][0], t))
                        test_tasks_to_discovered_programs[t] = test_tasks_to_discovered_programs.get(t, []) + [task_to_decoded_programs[t][p_idx]]

        # keep top-n programs for each task
        print("test_tasks_to_discovered_programs \n{}".format(test_tasks_to_discovered_programs))
        test_tasks_to_best_discovered_programs = {t: sorted(programs, key=lambda x: x[1].totalScore)[:TOP_N] for t,programs in test_tasks_to_discovered_programs.items()}
        print("test_tasks_to_best_discovered_programs (top {})\n{}".format(TOP_N, test_tasks_to_best_discovered_programs))
        # importantly we use tasks and not tasks_no_eval_ex so that ll is 0.0 only if programs get eval example correct as well
        response = execute_programs([t for t in tasks if t.name in test_tasks_to_best_discovered_programs], grammar, test_tasks_to_best_discovered_programs)
        test_tasks_to_lls = {}
        for item in response:
            test_tasks_to_lls[item["task"]] = item["log_likelihoods"]
        test_tasks_solved = [t for t in test_tasks_to_best_discovered_programs.keys() if any([ll == 0.0 for ll in test_tasks_to_lls[t]])]
        print("test tasks solved", test_tasks_solved)
        return


    else:

        for iteration in range(resume_iter, num_cycles):
     
            for start_idx, end_idx in get_batch_start_end_idxs(len(larc_train_dataset_cpu), batch_size):
                larc_train_dataset_batch_cpu = larc_train_dataset_cpu[start_idx:end_idx]
            
                while pretraining:
                    # train
                    num_epochs_start = 10 * num_epochs_start if fixed_epoch_pretrain == 0 else fixed_epoch_pretrain
                    task_to_programs = {k:v for k,v in task_to_programs.items() if ((len(v) > 0) and (k in tasks_subset))}
                    task_to_programs = {k:v[:(min(max_programs_per_task, len(v)))] for k,v in task_to_programs.items()}
                    if verbose:
                        for task, programs in task_to_programs.items():
                            print("\n\n{}: {}".format(task, "\n".join([" ".join([idx_to_token[i] for i in p[0]]) for p in programs]))) 
                    model = model.to(device=torch.device("cuda"))
                    model = train_experience_replay(model, pretrain_dataset_gpu, num_epochs=num_epochs_start, lr=lr, weight_decay=weight_decay, batch_size=sum([len(programs) for t,programs in task_to_programs.items()]))

                    # decode
                    # decode with randomized beam search
                    print("Starting to decode")
                    decode_start_time = time.time()
                    model = model.to(device=torch.device("cpu"))
                    task_to_decoded_programs, task_to_lls = multicore_decode(model, grammar, pretrain_dataset_cpu, tasks, restrict_types=restrict_types, rnn_decode=rnn_decode, num_iter_beam_search=num_iter_beam_search,
                    how="randomized_beam_search", beam_width=beam_width, epsilon=0.1, num_cpus=num_cpus, verbose=verbose)
                    print("\nFinished Decoding in {}s \n".format(time.time() - decode_start_time))

                    # check if all preloaded frontiers successfully decoded
                    tasks_solved = [t for t in task_to_decoded_programs.keys() if any([ll == 0.0 for ll in task_to_lls[t]])]
                    print("tasks_solved", tasks_solved)
                    
                    if fixed_epoch_pretrain or (len(tasks_solved) >= len(list(task_to_programs.keys()))):
                        pretraining = False
                        torch.save({
                           'model_state_dict': model.state_dict(),
                        }, '{}/model_done_pretraining.pt'.format(outputDirectory))

                if len(task_to_correct_programs_iter) > 0:
                    model = model.to(device=torch.device("cuda"))
                    iter_train_dataset = LARC_Cell_Dataset(tasks_dir, tasks_subset=list(task_to_correct_programs_iter.keys()), num_ios=MAX_NUM_IOS, resize=(30, 30), for_synthesis=True, beta=beta, task_to_programs=task_to_correct_programs_iter, device=device)
                    model = train_experience_replay(model, iter_train_dataset, num_epochs=epochs_per_replay, lr=lr, weight_decay=weight_decay, batch_size=TRAIN_BATCH_SIZE)
                
                    torch.save({
                       'model_state_dict': model.state_dict(),
                    }, '{}/model_{}.pt'.format(outputDirectory, iteration))
                    dill.dump(task_to_correct_programs, open("{}/task_to_correct_programs.pkl".format(outputDirectory), "wb"))
           
      
                # decode with randomized beam search
                print("Starting to decode")
                decode_start_time = time.time()
                model = model.to(device=torch.device("cpu"))
                task_to_decoded_programs, task_to_lls = multicore_decode(model, grammar, larc_train_dataset_batch_cpu, tasks, restrict_types=restrict_types, rnn_decode=rnn_decode, num_iter_beam_search=num_iter_beam_search,
                    how="randomized_beam_search", beam_width=beam_width, epsilon=epsilon, num_cpus=num_cpus, verbose=verbose)
                print("\nFinished Decoding in {}s \n".format(time.time() - decode_start_time))
                
                # experience replay train with discovered program
                task_to_correct_programs_iter = {}
                for task,log_likelihoods in task_to_lls.items():
                    programs_for_task = []
                    for i,ll in enumerate(log_likelihoods):
                        if ll == 0.0:
                            programName, program = task_to_decoded_programs[task][i]
                            print("Task {}: {} ({})".format(task, programName, program.totalScore))
     
                            paddedProgramTokenSeq = pad_token_seq(program.programTokenSeq, token_to_idx["PAD"], MAX_PROGRAM_LENGTH)
                            programScore = program.totalScore
                            if use_cuda:
                                # put tensor on gpu for training
                                programScore = torch.tensor(programScore, device=torch.device("cuda"))
                            programs_for_task.append((paddedProgramTokenSeq, programScore))

                            # add to library of discovered programs if it is not already there
                            task_programs = task_to_correct_programs.get(task, [])
                            existingProgramNames = [programName for programName, p in task_programs]
                            if programName not in existingProgramNames:
                                task_programs.append((programName, program))
                                task_to_correct_programs[task] = task_programs
                    
                    if len(programs_for_task) > 0:
                        if limit_overfit:
                            if not task_to_recently_decoded[task]:
                                task_to_correct_programs_iter[task] = programs_for_task
                        else:
                            task_to_correct_programs_iter[task] = programs_for_task
                        task_to_recently_decoded[task] = True
                    else:
                        task_to_recently_decoded[task] = False
                
                print("Training on discovered programs for {} tasks at iteration {}".format(len(task_to_correct_programs_iter), iteration))
                print("Decoded correct programs for {} tasks total".format(len(task_to_correct_programs)))
            
            print("Decoded correct program for {} tasks at iteration {}".format(len([el for el in list(task_to_recently_decoded.values()) if el]), iteration))
      
                # task_to_correct_programs_iter = {'1cf80156.json': [(torch.tensor([1, 25, 8, 45, 4, 79, 33, 8, 10, 11, 42, 4, 79, 79, 2, # 3, 2, 78, 78, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device=torch.device("cuda")), torch.tensor(2.8922, device=torch.device('cuda'))), #  (torch.tensor([1, 25, 8, 45, 4, 79, 33, 8, 10, 11, 42, 4, 79, 79, 2, 3, 2, 78, 78, 79, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device=torc # h.device('cuda')), torch.tensor(2.7261, device=torch.device('cuda')))]}

