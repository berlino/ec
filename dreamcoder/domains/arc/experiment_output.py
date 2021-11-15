import dill
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import subprocess

from dreamcoder.frontier import Frontier
from dreamcoder.grammar import Grammar
from dreamcoder.domains.arc.main import retrieveARCJSONTasks
from dreamcoder.domains.arc.arcPrimitives import basePrimitives, leafPrimitives, moreSpecificPrimitives
from dreamcoder.utilities import get_root_dir, numberOfCPUs

LARC_DIR = "data/larc/"
DATA_DIR = "data/arc"
TRAIN_TEST_SPLIT_FILENAME = "train_test_split.json"
LANGUAGE_ANNOTATIONS_FILE = os.path.join(DATA_DIR, "language/sentences/language.json") # All language annotations for training.
TOP_N = 3
FIGURE_PATH = "test_tasks_over_time_all_2.png"
NUM_MC_SAMPLES = 10000
EPSILON = 1.0 / float(NUM_MC_SAMPLES)
# EPSILON = 0.0
TASK_TO_PRIMITIVE_MARGINALS_FILENAME = "task_to_primitive_marginals_10000.pkl"
PLOT_FILENAME = "task_to_primitive_marginals_10000_log_odds_avg_epsilon={}.png".format(EPSILON)
# TASK_TO_PRIMITIVE_MARGINALS_FILENAME = None

# trained for 1000 recognition steps per iteration 
IO_BIGRAM_CHECKPOINT = "experimentOutputs/arc/2021-09-30T23:07:48.432156/arc_aic=1.0_arity=0_BO=True_CO=True_ES=1_ET=720_t_zero=1_HR=0.0_it=5_MF=10_noConsolidation=True_pc=10_RS=1000_RT=3600_RR=False_RW=False_solver=ocaml_STM=True_L=1.0_batch=200_TRR=randomShuffle_K=2_topkNotMAP=False_UET=3600_DSL=False_FTM=True.pickle"
IO_NL_BIGRAM_CHECKPOINT = "experimentOutputs/arc/2021-09-30T15:06:29.861706/arc_aic=1.0_arity=0_BO=True_CO=True_ES=1_ET=720_t_zero=1_HR=0.0_it=5_MF=10_noConsolidation=True_pc=10_RS=1000_RT=3600_RR=False_RW=False_solver=ocaml_STM=True_L=1.0_batch=200_TRR=randomShuffle_K=2_topkNotMAP=False_UET=3600_DSL=False_FTM=True.pickle"
IO_NL_PSEUDO_BIGRAM_CHECKPOINT = "experimentOutputs/arc/2021-10-01T11:24:05.422831/arc_aic=1.0_arity=0_BO=True_CO=True_ES=1_ET=720_t_zero=1_HR=0.5_it=5_MF=10_noConsolidation=True_pc=10_RS=1000_RT=3600_RR=False_RW=False_solver=ocaml_STM=True_L=1.0_batch=200_TRR=randomShuffle_K=2_topkNotMAP=False_UET=3600_DSL=False_FTM=True.pickle"

# trained for 10,000 recognition steps per iteration
IO_BIGRAM_CHECKPOINT = "experimentOutputs/arc/2021-09-30T14:45:18.915411/arc_aic=1.0_arity=0_BO=True_CO=True_ES=1_ET=720_t_zero=1_HR=0.0_it=5_MF=10_noConsolidation=True_pc=10_RS=10000_RT=3600_RR=False_RW=False_solver=ocaml_STM=True_L=1.0_batch=200_TRR=randomShuffle_K=2_topkNotMAP=False_UET=3600_DSL=False_FTM=True.pickle"
IO_NL_BIGRAM_CHECKPOINT = "experimentOutputs/arc/2021-09-30T14:38:38.039529/arc_aic=1.0_arity=0_BO=True_CO=True_ES=1_ET=720_t_zero=1_HR=0.0_it=5_MF=10_noConsolidation=True_pc=10_RS=10000_RT=3600_RR=False_RW=False_solver=ocaml_STM=True_L=1.0_batch=200_TRR=randomShuffle_K=2_topkNotMAP=False_UET=3600_DSL=False_FTM=True.pickle"
IO_NL_PSEUDO_BIGRAM_CHECKPOINT = "experimentOutputs/arc/2021-10-02T01:56:55.896457/arc_aic=1.0_arity=0_BO=True_CO=True_ES=1_ET=720_t_zero=1_HR=0.5_it=5_MF=10_noCons=True_pc=10_RS=10000_RT=3600_RR=False_RW=False_STM=True_L=1.0_batch=200_TRR=randomShuffle_K=2_topkNotMAP=False_UET=3600_DSL=False_FTM=True.pickle"
IO_CNN_NL_PSEUDO_BIGRAM_CHECKPOINT = "experimentOutputs/arc/2021-10-01T00:09:11.781052/arc_aic=1.0_arity=0_BO=True_CO=True_ES=1_ET=720_t_zero=1_HR=0.5_it=5_MF=10_noConsolidation=True_pc=10_RS=10000_RT=3600_RR=False_RW=False_solver=ocaml_STM=True_L=1.0_batch=200_TRR=randomShuffle_K=2_topkNotMAP=False_UET=3600_DSL=False_FTM=True.pickle"

NL_TEST_TIME = 0.00452113151550293

# testtasks solved with IO_NL_PSEUDO but not IO_BIGRAM
# ["0520fde7.json", "0b148d64.json", "1f85a75f.json",  "23b5c85d.json", "5582e5ca.json", "6f8cd79b.json", "72ca375d.json"]

class Result:
    def __init__(self, path, label):
        self.label = label
        self.result = dill.load(open(path, "rb"))
        train_task_names, test_task_names, trainTasksWithNl, testTasksWithNl, grammar, tasks, request = load_relevant_data()
        
        print("\n{}".format(self.label))
        try:
            self.task_to_program_to_solved = run_synthesized_programs_on_holdout(self.result, tasks, grammar, trainTasksWithNl, testTasksWithNl, train_task_names, test_task_names)
        except Exception as e:
            print(e)
            print("WARNING: Failed to initialize task_to_program_to_solved, can't check test enumeration results on holdout example")
            self.task_to_program_to_solved = {}

        self.test_tasks_solved = list(self.task_to_program_to_solved.keys())
        print("test tasks solved", self.test_tasks_solved)
        self.test_frontier_solutions = {t:Frontier([e for e in frontiers[-1].entries if self._is_solution(t, e.program)], t) for t,frontiers in self.result.frontiersOverTime.items() if t.name in testTasksWithNl}

    def _is_solution(self, task, program):
        if task.name in self.task_to_program_to_solved:
            program_string = str(program)
            if program_string in self.task_to_program_to_solved[task.name]:
                return self.task_to_program_to_solved[task.name][str(program)]
        return False



def taskMessage(t, task_to_programs):
    m = {
        "examples": [{"inputs": [xs[0].toJson()], "output": y.toJson()} for xs, y in t.examples],
        "name": t.name,
        "request": t.request.json(),
        "programs": [el[0] for el in task_to_programs[t.name]]
    }
    return m

def execute_programs(tasks, grammar, task_to_programs):

    message = {
        "tasks": [taskMessage(t, task_to_programs) for t in tasks],
        "programTimeout": 0.1,
    }
    dumped_message = json.dumps(message)
    with open('message', 'w') as outfile:
        json.dump(message, outfile) 

    try:
        solver_file = os.path.join(get_root_dir(), "solvers/exec_arc_p")
        process = subprocess.Popen(
            solver_file, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        response, error = process.communicate(bytes(dumped_message, encoding="utf-8"))
        response = json.loads(response.decode("utf-8"))
        return response
        
    except OSError as exc:
        raise exc

def best_first_enumeration(recognitionModel, grammar, tasks, testTasksWithNl, request):
     testTasks = [t for t in tasks if t.name in testTasksWithNl]
     allFrontiers, times = recognitionModel.enumerateFrontiers(testTasks, enumerationTimeout=30, CPUs=numberOfCPUs(), frontierSize=10, solver="dummy", maximumFrontier=TOP_N)
     task_name_to_time = {task.name:time for task,time in times.items()}

     test_tasks_to_programs = {frontier.task.name : [(str(e.program), None) for e in frontier.entries] for  frontier in allFrontiers}
     ocaml_execute_programs(grammar, testTasks, test_tasks_to_programs)
     print(task_name_to_time)
     return

def ocaml_execute_programs(grammar, tasks, test_tasks_to_programs):
     response = execute_programs([t for t in tasks if t.name in test_tasks_to_programs], grammar, test_tasks_to_programs)
     print(response)
     print("{} test tasks solved".format(len([r for r in response if any([ll == 0.0 for ll in r['log_likelihoods']])])))
     return

def run_synthesized_programs_on_holdout(result, tasks, grammar, trainTasksWithNl, testTasksWithNl, train_task_names, test_task_names):
     
     test_tasks_to_programs = {}
     for t,frontiers in result.frontiersOverTime.items():
         f = frontiers[-1]
         if len(f.entries) > 0:
             if t.name in train_task_names:
                 assert t.name in trainTasksWithNl
             elif t.name in test_task_names:
                 # None added for compatibility with exec_arc_p
                 top_n_entries = sorted([e for e in f.entries], reverse=True, key=lambda e: e.logPrior)[:TOP_N]
                 test_tasks_to_programs[t.name] = [(str(e.program), None) for e in top_n_entries]

     print("{} test tasks program found".format(len(test_tasks_to_programs)))
     test_tasks_to_programs_with_nl = {t:v for t,v in test_tasks_to_programs.items() if t in testTasksWithNl}
     print("{} test tasks with NL program found".format(len(test_tasks_to_programs_with_nl)))
     response = execute_programs([t for t in tasks if t.name in test_tasks_to_programs_with_nl], grammar, test_tasks_to_programs_with_nl)
     print("{} test tasks solved (with NL)".format(len([r for r in response if any([ll == 0.0 for ll in r['log_likelihoods']])])))
     task_to_program_to_solved = {r["task"]: {test_tasks_to_programs_with_nl[r["task"]][i][0]: (ll == 0.0) for i,ll in enumerate(r['log_likelihoods'])} for r in response}
     return {t: program_to_solved_dict for t,program_to_solved_dict in task_to_program_to_solved.items() if any(list(program_to_solved_dict.values()))}

def load_relevant_data():
    # load train and test task names
    train_test_split_dict = json.load(open(LARC_DIR + TRAIN_TEST_SPLIT_FILENAME, "r"))
    train_task_names = [t for t in train_test_split_dict["train"]]
    test_task_names = [t for t in train_test_split_dict["test"]]

    with open(LANGUAGE_ANNOTATIONS_FILE, "r") as f:
       language_annotations_data = json.load(f)

    trainTasksWithNl = [t for t in train_task_names if t in language_annotations_data]
    print("{} train tasks with NL".format(len(trainTasksWithNl)))
    testTasksWithNl = [t for t in test_task_names if t in language_annotations_data]
    print("{} test tasks with NL".format(len(testTasksWithNl)))

    grammar = Grammar.uniform(basePrimitives() + leafPrimitives() + moreSpecificPrimitives())
    # by including eval example in training we can execute using ocaml and see if program is consistent with holdout
    tasks = retrieveARCJSONTasks('arc_data/data/training', useEvalExamplesForTraining=True, filenames=None)
    request = tasks[0].request

    return train_task_names, test_task_names, trainTasksWithNl, testTasksWithNl, grammar, tasks, request

def plot_frontiers_single_iter(result, testTasksWithNl, label):
    assert len(result.result.testingSearchTime)
    times = [time for task,time in result.result.testSearchTime.items() if task.name in result.test_tasks_solved]
    sorted_times = sorted(times)
    # plt.ylim(0, 183)
    plt.xlim(0, 720)
    plt.xlabel("time (s)", fontsize=12)
    plt.ylabel("number of testing tasks solved (183 total)", fontsize=12)
    
    plt.plot([0] + sorted_times + [720], list(range(len(sorted_times)+1)) + [len(sorted_times)], label=label)
    return


def show_test_programs_found(result_a, result_b):

    for t,f_a in result_a.test_frontier_solutions.items():
        f_b = result_b.test_frontier_solutions[t]
        heading = "\nTask: {}\n{}\n--------------------------------------------------".format(t.name, t.sentences)
        to_print = []
        if not f_a.empty:
            to_print.append(heading)
            to_print.append("{}: {} ({})".format(result_a.label, f_a.topK(1).entries[0].program, f_a.topK(1).entries[0].logPrior))
            if not f_b.empty:
                to_print.append("{}: {} ({})\n".format(result_b.label, f_b.topK(1).entries[0].program, f_b.topK(1).entries[0].logPrior))
        elif not f_b.empty:
            to_print.append(heading)
            to_print.append("{}: {} ({})".format(result_b.label, f_b.topK(1).entries[0].program, f_b.topK(1).entries[0].logPrior))
        
        if len(to_print) > 0:
            print("\n".join(to_print))
    return

def _plot_helper(sorted_log_odds):
    plt.rcParams["figure.figsize"] = (12,5)
    plt.bar(x=range(len(sorted_log_odds)), height=[el[1] for el in sorted_log_odds], tick_label=[el[0] for el in sorted_log_odds])
    if EPSILON == 0.0:
        plt.xticks(rotation=90, fontsize=7)
    else:
        plt.xticks(rotation=90, fontsize=9)
    plt.ylabel("odds (log scale)")
    plt.tight_layout()
    plt.savefig(PLOT_FILENAME, dpi=1200)
    plt.show()

def _plot_relative_primitive_marginals_task(task_to_primitive_marginals, task_name):

    log_odds = [(el[0], np.log(el[1] + EPSILON) - np.log(el[2] + EPSILON)) for el in task_to_primitive_marginals[task_name]]
    sorted_log_odds = sorted(log_odds, key=lambda pair: pair[1], reverse=True)
    _plot_helper(sorted_log_odds)
    return

def _plot_relative_primitive_marginals_tasks(task_to_primitive_marginals):
    
    # key is primitive, value is (list of primitive probs for model_a, list of primitive probs for model_b)
    primitive_to_probs = {}
    for t,data in task_to_primitive_marginals.items():
        for el in data:
            primitive_to_probs[el[0]] = primitive_to_probs.get(el[0], ([], []))
            # append probability of prim from model_a
            primitive_to_probs[el[0]][0].append(el[1])
            # append probability of prim from model_b
            primitive_to_probs[el[0]][1].append(el[2])
    
    print("\nprimitive_to_probs\n{}".format(primitive_to_probs))

    if EPSILON == 0.0:
        avg_log_odds = [(p + " ({})".format(len(primitive_to_probs[p][0])), np.log(np.mean(np.array(primitive_to_probs[p][0])) + EPSILON) - np.log(np.mean(np.array(primitive_to_probs[p][1])) + EPSILON)) 
            for p in primitive_to_probs.keys() if (np.mean(np.array(primitive_to_probs[p][0])) > 0.0 and np.mean(np.array(primitive_to_probs[p][1])) > 0.0)]
    else:
        avg_log_odds = [(p + " ({})".format(len(primitive_to_probs[p][0])), np.log(np.mean(np.array(primitive_to_probs[p][0])) + EPSILON) - np.log(np.mean(np.array(primitive_to_probs[p][1])) + EPSILON))
            for p in primitive_to_probs.keys()]
    # average log odds across tasks for each primitive
    sorted_log_odds = sorted(avg_log_odds, key=lambda pair: pair[1], reverse=True)
    print("sorted_log_odds\n {}\n".format(sorted_log_odds))
    _plot_helper(sorted_log_odds)
    return

def conditional_primitive_marginals(result_a, result_b, use_best=False, num_mc_samples=10):

    # first element in tuple is best program for result_a, second element in tuple is best_program for result_b
    task_to_programs_tuple = {}
    # first element in tuple is conditional bigram for result_a, second element in tuple is conditional bigram for result_b
    task_to_grammars_tuple = {}
    task_to_primitive_marginals = {}
    for t,f_a in result_a.test_frontier_solutions.items():
        
        f_b = result_b.test_frontier_solutions[t]
        # one of the models needs to have found a program for us to be able to do analysis
        if t.name in result_a.test_tasks_solved or t.name in result_b.test_tasks_solved:
            g_a = result_a.result.recognitionModel.grammarOfTask(t)
            g_b = result_b.result.recognitionModel.grammarOfTask(t)
            task_to_grammars_tuple[t] = (g_a, g_b)
            
            if not f_a.empty:
                best_program_a = f_a.topK(1).entries[0].program
            else:
                if use_best:
                    best_program_a = sorted([(e.program, g_a.logLikelihood(t.request, e.program)) for e in f_b.entries], key=lambda pair: pair[1], reverse=True)[0][0]
                else:
                    best_program_a = f_b.topK(1).entries[0].program
    
            if not f_b.empty:
                best_program_b = f_b.topK(1).entries[0].program
            else:
                if use_best:
                    best_program_b = sorted([(e.program, g_b.logLikelihood(t.request, e.program)) for e in f_a.entries], key=lambda pair: pair[1], reverse=True)[0][0]
                else:
                    best_program_b = f_a.topK(1).entries[0].program

            # for primitives that are shared in both best programs, calculate their marginals under the two different models
            # shared_primitives = list(set(best_program_a.left_order_tokens()).intersection(set(best_program_b.left_order_tokens())))
            # print(shared_primitives)
            # assert best_program_a == best_program_b
            shared_primitives = best_program_a.left_order_tokens()
            print("{}: {}".format(t.name, t.sentences))
            
            expected_uses_a, primitive_to_index = g_a.marginalsMonteCarlo(t.request, returnPrimitive2index=True, numMcSamples=num_mc_samples) 
            expected_uses_b, primitive_to_index = g_b.marginalsMonteCarlo(t.request, returnPrimitive2index=True, numMcSamples=num_mc_samples)

            primitive_marginals = [[primitive, expected_uses_a[primitive_to_index[primitive]], expected_uses_b[primitive_to_index[primitive]]] for primitive in shared_primitives]
            print("primitive marginals\n{}\n".format(primitive_marginals))
            task_to_primitive_marginals[t.name] = primitive_marginals
        else:
            continue

    print(task_to_primitive_marginals)
    dill.dump(task_to_primitive_marginals, open("task_to_primitive_marginals_{}.pkl".format(num_mc_samples), "wb"))
    return task_to_primitive_marginals 

def experiment_output_main(action):

    paths = [IO_BIGRAM_CHECKPOINT, IO_NL_BIGRAM_CHECKPOINT, IO_NL_PSEUDO_BIGRAM_CHECKPOINT, IO_CNN_NL_PSEUDO_BIGRAM_CHECKPOINT]
    labels = ["IO", "IO + NL", "IO + NL (pseudo)", "IO (CNN) + NL (pseudo)"]
    results = {label: Result(path, label) for label,path in zip(labels, paths)}
    
    train_task_names, test_task_names, trainTasksWithNl, testTasksWithNl, grammar, tasks, request = load_relevant_data()

    if action == "plot":
        plt.plot([0, NL_TEST_TIME, 720], [0, 1, 1], label="NL")
        plt.plot([0,720], [0,0], label="NL (pseudo)")
        for label,result in results.items():
            plot_frontiers_single_iter(result, testTasksWithNl, label)
        plt.legend(fontsize=12)
        plt.savefig(FIGURE_PATH, dpi=1200)
        plt.show()

    elif action == "best_first":
        best_first_enumeration(results["IO + NL"].result.recognitionModel, grammar, tasks, testTasksWithNl, request)

    elif action == "show":
        show_test_programs_found(results["IO + NL (pseudo)"], results["IO"])

    elif action == "plot_marginals":
        if TASK_TO_PRIMITIVE_MARGINALS_FILENAME is not None:
            task_to_primitive_marginals = dill.load(open(TASK_TO_PRIMITIVE_MARGINALS_FILENAME, "rb"))
        else:
            task_to_primitive_marginals = conditional_primitive_marginals(results["IO + NL (pseudo)"], results["IO"], num_mc_samples=NUM_MC_SAMPLES)
        _plot_relative_primitive_marginals_tasks(task_to_primitive_marginals)
    return
