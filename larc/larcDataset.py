import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import os
import json

from dreamcoder.program import Program
from larc.decoderUtils import program_to_token_sequence

PAD_VAL = 10

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

def load_task_to_programs_from_frontiers_json(grammar, token_to_idx, json_file_name="data/arc/prior_enumeration_frontiers_8hr.json"):
    """
    Load prior enumeration frontiers and process into dictionary with task names as keys and lists of corresponding programs as values.
    Each program is represented as a list of indicies created used token_to_idx argument.
    """

    task_to_programs_raw = json.load(open(json_file_name, 'r'))
    task_to_programs = {}
    for task, task_programs in task_to_programs_raw.items():
        task_to_programs[task] = []
        for program_string in task_programs:

            # hacky way to check that all programs to be used for imitation learning can be parsed (e.g. don't contain primitives not in our grammar)
            try: 
                program = Program.parse(program_string)
            except:
                continue

            task_to_programs[task].append([token_to_idx[token] for token in program_to_token_sequence(program, grammar)])
    return task_to_programs


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

    def __init__(self, tasks_json_path, resize=(30,30), num_ios=3, tasks_subset=None, max_tasks=float('inf'), task_to_programs=None, device=torch.device("cpu")):
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

        task_generator = self.gen_larc_pred_tasks(tasks_json_path, tasks_subset) if task_to_programs is None \
        else self.gen_larc_synth_tasks(tasks_json_path, tasks_subset, self.task_to_programs)

        # pad all grids with 0s to make same size
        for i, larc_pred_task in enumerate(task_generator):

            # only load max_tasks tasks
            if i >= max_tasks:
                break

            new_task = larc_pred_task.copy()

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
            if num_ios is not None and len(new_ios) < num_ios:
                new_ios += [(arc2torch(np.full((30, 30), PAD_VAL), device=device),
                             arc2torch(np.full((30, 30), PAD_VAL), device=device)) for _ in range(num_ios - len(new_ios))]
            new_task['io_grids'] = new_ios

            # pad test input
            test_in = np.array(larc_pred_task['test_in'])
            test_in_size = resize if resize is not None else test_in.shape
            test_in_padded = np.full(test_in_size, PAD_VAL)
            test_in_padded[:test_in.shape[0], :test_in.shape[1]] = test_in
            new_task['test_in'] = arc2torch(test_in_padded, device=device)

            # tokenize description
            new_task['desc_tokens'] = {k: torch.tensor(v, device=device) for k, v in tokenizer.encode_plus(larc_pred_task['desc']).items()}

            # if we are generating tasks for synthesis model then we don't need x and y positions as input
            if task_to_programs is None:

                # 1-hot x and y
                max_x, max_y = 30, 30
                new_task['x'] = torch.zeros(max_x, device=device)
                new_task['x'][larc_pred_task['x']] = 1
                new_task['y'] = torch.zeros(max_y, device=device)
                new_task['y'][larc_pred_task['y']] = 1

            else:
                new_task["programs"] = [torch.tensor(token_sequence, device=device) for token_sequence in new_task["programs"]]

            # for key, value in new_task.items():
            #    print_device(value)            

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

    def gen_larc_synth_tasks(self, tasks_json_path, tasks_subset, task_to_programs):
        """
        generate larc tasks to train synthesis model with
        :param tasks_json_path: path to folder with LARC tasks
        :yields: {'io_grids': [(input1, output1), (input2, output2)...], 'test_in': test_input, 'desc': NL description, 'programs': list of programs (strings)}
        """

        for base_task in self.gen_larc_tasks(tasks_json_path, tasks_subset=tasks_subset):  # {'io_grids': [(input1, output1), (input2, output2)...], 'test': (test_input, test output), 'desc': NL description}
            for task in self.augment_larc_task(base_task):
                test_in, test_out = task['test']
                if len(task_to_programs[task['name']]) == 0:
                    continue
                else:
                    yield {'io_grids': task['io_grids'], 'test_in': test_in, 'desc': task['desc'], 'num': task['num'], 'name': task['name'], 'programs': task_to_programs[task['name']]}


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

    def gen_larc_tasks(self, task_json_path, min_perc=0.1, tasks_subset=None):
        """
        generator for tasks for input to NN
        :param task_json_path: path to folder with LARC tasks
        :min_perc minimum fraction of successful communications to include description in dataset
        :yields: {'io_grids': [(input1, output1), (input2, output2)...], 'test': (test_input, test output), 'desc': NL description}
        """

        num_tasks = 0
        for fname in os.listdir(task_json_path):

            task_num = int(fname.split('.')[0])

            # if subset specified, ignore tasks not in subset
            if tasks_subset is not None and task_num not in tasks_subset:
                continue

            with open(os.path.join(task_json_path, fname), 'r') as f:
                task = json.load(f)
                io_exs = []

                # get examples IOs
                for t in task['train']:
                    io_exs.append((t['input'], t['output']))

                # get test IO
                io_test = (task['test'][0]['input'], task['test'][0]['output'])

                # yield for each description
                for desc in task['descriptions'].values():
                    suc, tot = 0, 0
                    for build in desc['builds'].values():
                        suc += 1 if build['success'] else 0
                        tot += 1
                    if tot > 0 and suc / tot >= min_perc:   # tot > 0 to avoid / 0
                        num_tasks += 1
                        yield {'io_grids': io_exs, 'test': io_test, 'desc': desc['do_description'], 'num': task_num, 'name': task['name']}
