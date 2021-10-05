import dill
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import os
import json

from dreamcoder.program import Program
from neural_seq.decoderUtils import program_to_token_sequence

PAD_VAL = 10
TOKEN_PAD_VALUE = -1
MAX_DESC_SEQ_LENGTH = 350
MAX_NUM_IOS = 5


def get_batch_start_end_idxs(n, batch_size):
    for i in range(0, n, batch_size):
        yield i, min(i+batch_size, n)


def normalize_weights(programs, beta):
    """
    :param programs: list of (token_idx_seq, weight) tuples. each weight is the negative log probability of each program (and not the probability)
    :param beta: controls how much to rely on weights, with beta=0 corresponding to uniform distribution and beta=1 corresponding to leaving as is
    """

    denominator = sum([torch.exp(-w)**beta for _,w in programs])
    return [(p, torch.exp(-w)**beta / denominator) for p,w in programs]

def collate(x, inlcude_ground_truth_programs):

    def stack_entry(x, name):
        return torch.stack([x[i][name] for i in range(len(x))])

    # stack all tensors of the same input/output type and the same example index to form batch
    io_grids_batched = [(torch.stack([x[i]["io_grids"][ex_idx][0] for i in range(len(x))]), torch.stack([x[i]["io_grids"][ex_idx][1] for i in range(len(x))])) 
        for ex_idx in range(MAX_NUM_IOS)]
    
    batch_data = {
                "name": [x[i]["name"] for i in range(len(x))],
                "io_grids": io_grids_batched,
                "test_in": stack_entry(x, "test_in"), 
                "desc_tokens": {key: torch.stack([x[i]["desc_tokens"][key] for i in range(len(x))]) for key in x[0]["desc_tokens"].keys()}}

    if inlcude_ground_truth_programs:
        batch_data["program"] = stack_entry(x, "program")
        batch_data["program_weight"] = stack_entry(x, "program_weight")
    
    return batch_data

def onehot_initialization(a, num_cats):
    """https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy"""
    out = np.zeros((a.size, num_cats), dtype=np.uint8)  # initialize correct size 3-d tensor of 0s
    out[np.arange(a.size),a.ravel()] = 1
    out.shape = a.shape + (num_cats,)
    return out


def arc2torch(grid, device, num_cats=11):
    """convert 2-d grid of original arc format to 3-d one-hot encoded tensor"""
    grid = onehot_initialization(grid, num_cats)
    grid = np.rollaxis(grid, 2)
    return torch.from_numpy(grid).float().to(device)

def print_device(el):
    if type(el) == torch.Tensor:
        print(el.device)
    elif type(el) == list or type(el) == tuple:
        for x in el:
            print_device(x)
    else:
        print("type of el is: {}".format(type(el)))
    return

def pad_token_seq(token_sequence, pad_token, max_program_length):
    # pad on the right so that all token sequences are the same length
    while len(token_sequence) < max_program_length:
        token_sequence.append(pad_token)
    return token_sequence

def preload_frontiers_to_task_to_programs(preload_frontiers_filename):

    with open(preload_frontiers_filename, "rb") as handle:
        result = dill.load(handle)
    preloaded_frontiers = result.allFrontiers
    
    print("preloaded frontiers", preloaded_frontiers)
    task_to_programs = {}
    for t,frontier in preloaded_frontiers.items():
        if not frontier.empty:
            task_to_programs[t.name] = [str(e.program) for e in frontier.entries]
    print(task_to_programs)
    return task_to_programs

def process_task_to_programs(grammar, token_to_idx, max_program_length, task_to_programs, device):
    """
    Load prior enumeration frontiers and process into dictionary with task names as keys and lists of corresponding programs as values.
    Each program is represented as a list of indicies created used token_to_idx argument. Pads programs so they are all the same length.
    """
    processed_task_to_programs = {}
    for task, task_programs in task_to_programs.items():
        processed_task_to_programs[task] = []
        for program_string in task_programs:

            # hacky way to check that all programs to be used for imitation learning can be parsed (e.g. don't contain primitives not in our grammar)
            try: 
                program = Program.parse(program_string)
            except:
                continue
            # seq
            token_sequence = [token_to_idx[token] for token in program_to_token_sequence(program, grammar)]
            padded_token_sequence = pad_token_seq(token_sequence, token_to_idx["PAD"], max_program_length)
            # append token sequence and the score of the program. Default to 1.0 since we want to equallly weight all frontier entries
            processed_task_to_programs[task].append((padded_token_sequence, torch.tensor(1.0, device=device, requires_grad=False)))
    return processed_task_to_programs


def load_task_to_programs_from_frontiers_pkl(grammar, request, token_to_idx, pkl_name="data/arc/prior_enumeration_frontiers_8hr.pkl"):
    """
    Load prior enumeration frontiers and process into dictionary with task names as keys and lists of corresponding programs as values.
    Each program is represented as a list of indicies created used token_to_idx argument.
    """

    task_to_frontiers = pickle.load(open(pkl_name, 'rb'))
    assert all([task.request == request for task in task_to_frontiers.keys()])

    task_to_programs = {}
    for task, frontier in task_to_frontiers.items():
        all_programs_for_task = []

        # hacky way to check that all programs to be used for imitation learning can be parsed (e.g. don't contain primitives not in our grammar)
        for e in frontier.entries:
            try: 
                Program.parse(str(e.program))
                all_programs_for_task.append(e.program)
            except:
                pass

        task_to_programs[str(task)] = [[token_to_idx[token] for token in program_to_token_sequence(p, grammar)] for p in all_programs_for_task]
    return task_to_programs

class LARC_Cell_Dataset(Dataset):
    """dataset for predicting each cell color in LARC dataset."""

    def __init__(self, tasks_json_path, resize=(30,30), num_ios=3, tasks_subset=None, max_tasks=float('inf'), for_synthesis=False, beta=0.0, 
        task_to_programs=None, device=torch.device("cpu"), task_to_sentences=None):
        """
        Params:
            tasks_json_path: path to folder with task jsons in it
            resize: grid size to pad each grid with so uniform size. If None, will not pad.
            num_ios: number of IO examples to return per task. If task has less, will pad with empty task. If task has
                    more, will take first num_ios tasks. If None, will just return all IO examples.
            tasks_subset: list of tasks to include in dataset. If None, uses all
            max_tasks: maximum number of tasks to load
        """
        self.tasks = []
        self.task_to_programs = task_to_programs
        tasks_subset = set(tasks_subset) if tasks_subset is not None else None    # for O(1) checks
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir=".cache/")

        task_generator = self.gen_larc_synth_tasks(tasks_json_path, tasks_subset, self.task_to_programs, beta=beta, 
            task_to_sentences=task_to_sentences) if for_synthesis \
        else self.gen_larc_pred_tasks(tasks_json_path, tasks_subset)

        # pad all grids with 0s to make same size
        for i, larc_pred_task in enumerate(task_generator):

            # only load max_tasks tasks
            if i >= max_tasks:
                break

            new_task = larc_pred_task.copy()

            # if we are generating tasks for synthesis model then we don't need x and y positions as input
            if for_synthesis:
                if task_to_programs is not None:
                    if isinstance(new_task["program"], list):
                        new_task["program"] = torch.tensor(new_task["program"], device=device)
            else:
                # 1-hot x and y
                max_x, max_y = 30, 30
                new_task['x'] = torch.zeros(max_x, device=device)
                new_task['x'][larc_pred_task['x']] = 1
                new_task['y'] = torch.zeros(max_y, device=device)
                new_task['y'][larc_pred_task['y']] = 1

            # pad IOs
            new_ios = []
            io_exs = larc_pred_task['io_grids'][:num_ios] if num_ios is not None else larc_pred_task['io_grids']
            for io_in, io_out in io_exs:
                io_in = np.array(io_in)
                in_size = resize if resize is not None else io_in.shape
                io_in_padded = np.full(in_size, PAD_VAL)
                io_in_padded[:io_in.shape[0],:io_in.shape[1]] = io_in

                io_out = np.array(io_out)
                out_size = resize if resize is not None else io_out.shape
                io_out_padded = np.full(out_size, PAD_VAL)
                io_out_padded[:io_out.shape[0],:io_out.shape[1]] = io_out

                # make grid one-hot
                new_ios.append((arc2torch(io_in_padded, device=device),
                                arc2torch(io_out_padded, device=device)))

            # TODO: mask extra ios
            # ensure same number IO examples per task (give 1x1 placeholder task)
            if num_ios is not None:
                while len(new_ios) < num_ios:
                    new_ios.append((arc2torch(np.full((30, 30), PAD_VAL), device=device),arc2torch(np.full((30, 30), PAD_VAL), device=device)))
            new_task['io_grids'] = new_ios

            # pad test input
            test_in = np.array(larc_pred_task['test_in'])
            test_in_size = resize if resize is not None else test_in.shape
            test_in_padded = np.full(test_in_size, PAD_VAL)
            test_in_padded[:test_in.shape[0], :test_in.shape[1]] = test_in
            new_task['test_in'] = arc2torch(test_in_padded, device=device)

            # tokenize description
            # padding all sequences to max length of MAX_DESC_SEQ_LENGTH tokens to make batching easier
            new_task['desc_tokens'] = {k: torch.tensor(v, device=device) for k, v in tokenizer.encode_plus(larc_pred_task['desc'], 
                padding='max_length', max_length=MAX_DESC_SEQ_LENGTH, pad_to_max_length=True).items()}

            self.tasks.append(new_task)

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.tasks[idx]

    def augment_larc_task(self, task):
        """
        generator to augment tasks
        :param task: {'io_grids': [(input1, output1), (input2, output2)...], 'test': (test_input, test output), 'desc': NL description}
        :yields: each augmented task
        """
        yield task

    def gen_larc_synth_tasks(self, tasks_json_path, tasks_subset, task_to_programs, beta, task_to_sentences):
        """
        generate larc tasks to train synthesis model with
        :param tasks_json_path: path to folder with LARC tasks
        :yields: {'io_grids': [(input1, output1), (input2, output2)...], 'test_in': test_input, 'desc': NL description, 'programs': list of programs (strings)}
        """

        for base_task in self.gen_larc_tasks(tasks_json_path, tasks_subset=tasks_subset, task_to_sentences=task_to_sentences):  # {'io_grids': [(input1, output1), (input2, output2)...], 'test': (test_input, test output), 'desc': NL description}
            for task in self.augment_larc_task(base_task):
                test_in, test_out = task['test']
                task_dict = {'io_grids': task['io_grids'], 'test_in': test_in, 'desc': task['desc'], 'num': task['num'], 'name': task['name']}
                if task_to_programs is not None:
                    programs = task_to_programs[task['name']]
                    weight_normalized_programs = normalize_weights(programs, beta)
                    print("weight-normalized-programs ({}): {}".format(task['name'], weight_normalized_programs))
                    for i in range(len(weight_normalized_programs)):
                        task_dict['program'] = weight_normalized_programs[i][0]
                        task_dict['program_weight'] = weight_normalized_programs[i][1].detach()
                        yield task_dict
                else:
                    yield task_dict


    def gen_larc_pred_tasks(self, tasks_json_path, tasks_subset):
        """
        generate prediction tasks for larc tasks, predicting the color for each x, y, cell in the test grid
        :param tasks_json_path: path to folder with LARC tasks
        :yields: {'io_grids': [(input1, output1), (input2, output2)...], 'test_in': test_input, 'desc': NL description, 'x': x, 'y': y, 'col': cell_color}
        """

        for base_task in self.gen_larc_tasks(tasks_json_path, tasks_subset=tasks_subset):  # {'io_grids': [(input1, output1), (input2, output2)...], 'test': (test_input, test output), 'desc': NL description}
            for task in self.augment_larc_task(base_task):
                test_in, test_out = task['test']

                # create prediction task for each coordinate
                for y, row in enumerate(test_out):
                    for x, cell_color in enumerate(row):
                        yield {'io_grids': task['io_grids'], 'test_in': test_in, 'desc': task['desc'], 'x': x, 'y': y,
                               'col': cell_color, 'num': task['num']}

    def gen_larc_tasks(self, task_json_path, min_perc=0.1, tasks_subset=None, task_to_sentences=None):
        """
        generator for tasks for input to NN
        :param task_json_path: path to folder with LARC tasks
        :min_perc minimum fraction of successful communications to include description in dataset
        :yields: {'io_grids': [(input1, output1), (input2, output2)...], 'test': (test_input, test output), 'desc': NL description}
        """

        num_tasks = 0
        for fname in os.listdir(task_json_path):

            task_num = int(fname.split('.')[0])

            with open(os.path.join(task_json_path, fname), 'r') as f:
                task = json.load(f)
                io_exs = []

            # if subset specified, ignore tasks not in subset
            if tasks_subset is not None and not(task["name"] in tasks_subset):
                continue
            else:

                # get examples IOs
                for t in task['train']:
                    io_exs.append((t['input'], t['output']))

                # get test IO
                io_test = (task['test'][0]['input'], task['test'][0]['output'])

                # If this file has been provided use its NL instead of the ones from task_json_path (hacky way to ensure we use same data
                # as bigram synthesis model)
                if task_to_sentences is not None:
                    if task["name"] in task_to_sentences:
                        description = ". ".join(task_to_sentences[task["name"]])
                    else:
                        # copying how we handle no language in Bigram model experiments. Encoder should learn to ignore "" for tasks we solve with no language
                        description = ""
                    yield {'io_grids': io_exs, 'test': io_test, 'desc': description, 'num': task_num, 'name': task['name']}

                else:
                    # yield for each description
                    for desc in task['descriptions'].values():
                        suc, tot = 0, 0
                        for build in desc['builds'].values():
                            suc += 1 if build['success'] else 0
                            tot += 1
                        if tot > 0 and suc / tot >= min_perc:   # tot > 0 to avoid / 0
                            num_tasks += 1
                            yield {'io_grids': io_exs, 'test': io_test, 'desc': desc['do_description'], 'num': task_num, 'name': task['name']}
