import dill
import json
import os
import matplotlib.pyplot as plt
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
FIGURE_PATH = "test_tasks_over_time.png"

# trained for 1000 recognition steps per iteration 
IO_BIGRAM_CHECKPOINT = "experimentOutputs/arc/2021-09-30T23:07:48.432156/arc_aic=1.0_arity=0_BO=True_CO=True_ES=1_ET=720_t_zero=1_HR=0.0_it=5_MF=10_noConsolidation=True_pc=10_RS=1000_RT=3600_RR=False_RW=False_solver=ocaml_STM=True_L=1.0_batch=200_TRR=randomShuffle_K=2_topkNotMAP=False_UET=3600_DSL=False_FTM=True.pickle"
IO_NL_BIGRAM_CHECKPOINT = "experimentOutputs/arc/2021-09-30T15:06:29.861706/arc_aic=1.0_arity=0_BO=True_CO=True_ES=1_ET=720_t_zero=1_HR=0.0_it=5_MF=10_noConsolidation=True_pc=10_RS=1000_RT=3600_RR=False_RW=False_solver=ocaml_STM=True_L=1.0_batch=200_TRR=randomShuffle_K=2_topkNotMAP=False_UET=3600_DSL=False_FTM=True.pickle"
IO_NL_PSEUDO_BIGRAM_CHECKPOINT = "experimentOutputs/arc/2021-10-01T11:24:05.422831/arc_aic=1.0_arity=0_BO=True_CO=True_ES=1_ET=720_t_zero=1_HR=0.5_it=5_MF=10_noConsolidation=True_pc=10_RS=1000_RT=3600_RR=False_RW=False_solver=ocaml_STM=True_L=1.0_batch=200_TRR=randomShuffle_K=2_topkNotMAP=False_UET=3600_DSL=False_FTM=True.pickle"

# trained for 10,000 recognition steps per iteration
IO_BIGRAM_CHECKPOINT = "experimentOutputs/arc/2021-09-30T14:45:18.915411/arc_aic=1.0_arity=0_BO=True_CO=True_ES=1_ET=720_t_zero=1_HR=0.0_it=5_MF=10_noConsolidation=True_pc=10_RS=10000_RT=3600_RR=False_RW=False_solver=ocaml_STM=True_L=1.0_batch=200_TRR=randomShuffle_K=2_topkNotMAP=False_UET=3600_DSL=False_FTM=True.pickle"
IO_NL_BIGRAM_CHECKPOINT = "experimentOutputs/arc/2021-09-30T14:38:38.039529/arc_aic=1.0_arity=0_BO=True_CO=True_ES=1_ET=720_t_zero=1_HR=0.0_it=5_MF=10_noConsolidation=True_pc=10_RS=10000_RT=3600_RR=False_RW=False_solver=ocaml_STM=True_L=1.0_batch=200_TRR=randomShuffle_K=2_topkNotMAP=False_UET=3600_DSL=False_FTM=True.pickle"
IO_NL_PSEUDO_BIGRAM_CHECKPOINT = "experimentOutputs/arc/2021-10-02T01:56:55.896457/arc_aic=1.0_arity=0_BO=True_CO=True_ES=1_ET=720_t_zero=1_HR=0.5_it=5_MF=10_noCons=True_pc=10_RS=10000_RT=3600_RR=False_RW=False_STM=True_L=1.0_batch=200_TRR=randomShuffle_K=2_topkNotMAP=False_UET=3600_DSL=False_FTM=True.pickle"

class Result:
    def __init__(self, path, label):
        self.label = label
        self.result = dill.load(open(path, "rb"))
        train_task_names, test_task_names, trainTasksWithNl, testTasksWithNl, grammar, tasks, request = _load_relevant_data()
        
        print("\n{}".format(self.label))
        try:
            self.task_to_program_to_solved = run_synthesized_programs_on_holdout(self.result, tasks, grammar, trainTasksWithNl, testTasksWithNl, train_task_names, test_task_names)
        except:
            print("WARNING: Failed to initialize task_to_program_to_solved, can't check test enumeration results on holdout example")
            self.task_to_program_to_solved = {}
        
        self.test_frontier_solutions = {t:Frontier([e for e in frontiers[-1].entries if self._is_solution(t, e.program)], t) for t,frontiers in self.result.frontiersOverTime.items()}


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
     allFrontiers, _ = recognitionModel.enumerateFrontiers(testTasks, enumerationTimeout=30, CPUs=numberOfCPUs(), frontierSize=10, solver="dummy", maximumFrontier=TOP_N)
     solved = set()

     test_tasks_to_programs = {frontier.task.name : [(str(e.program), None) for e in frontier.entries] for  frontier in allFrontiers}
     ocaml_execute_programs(grammar, testTasks, test_tasks_to_programs)
     return

def ocaml_execute_programs(grammar, tasks, test_tasks_to_programs):
     response = execute_programs([t for t in tasks if t.name in test_tasks_to_programs], grammar, test_tasks_to_programs)
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

     return {r["task"]: {test_tasks_to_programs_with_nl[r["task"]][i][0]: (ll == 0.0) for i,ll in enumerate(r['log_likelihoods'])} for r in response}

def _load_relevant_data():
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
    assert len(result.testingSearchTime)
    times = [time for task,time in result.testSearchTime.items() if task.name in testTasksWithNl]

    sorted_times = sorted(times)
    # plt.ylim(0, 50)
    # plt.xlim(0, 40)
    plt.xlabel("Time (s)")
    plt.ylabel("Number of tasks solved")
    plt.plot(sorted_times, range(len(sorted_times)), label=label)
    return


def get_expected_uses(result_a, result_b, testTasksWithNl):

    
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
 
    """
    for t,v in result_1.recognitionTaskMetrics.items():
            io_print.append(            to_print.append("Task: {}".format(t.name))
        if t.name in testTasksWithNl:
            grammar = result_1.recognitionModel.grammarOfTask(t)
            uses = grammar.expectedUsesMonteCarlo(t.request, debug=None)
            print(uses)
            return
    print(result.recognitionModel)
    """
    return

def experiment_output_main(action):

    paths = [IO_BIGRAM_CHECKPOINT, IO_NL_BIGRAM_CHECKPOINT, IO_NL_PSEUDO_BIGRAM_CHECKPOINT]
    labels = ["IO", "IO + NL", "IO + NL (pseudo)"]
    results = {label: Result(path, label) for label,path in zip(labels, paths)}
    
    train_task_names, test_task_names, trainTasksWithNl, testTasksWithNl, grammar, tasks, request = _load_relevant_data()

    if action == "plot":
        for label,result in results.items():
            plot_frontiers_single_iter(result.result, testTasksWithNl, label)
        plt.legend()
        plt.savefig(FIGURE_PATH)
        plt.show()

    elif action == "best_first":
        best_first_enumeration(results["IO + NL (pseudo)"].result.recognitionModel, grammar, tasks, testTasksWithNl, request)

    elif action == "conditional_bigrams":
        get_expected_uses(results["IO + NL (pseudo)"].result, results["IO"].result, testTasksWithNl)
    return
