from dreamcoder.domains.arc.utilsPostProcessing import *
from dreamcoder.program import Program
from dreamcoder.type import Context
from dreamcoder.utilities import get_root_dir

import json
import subprocess

class Stack:
    """
    Stack data structure to enforce type constraints when decoding a program by sequentially sampling
    its primitives
    """

    def __init__(self):
        self.stack = []

    def pop(self):
        return self.stack.pop(-1)

    def push(self, tp):
        self.stack.append(tp)
        return

    def toPop(self):
        if len(self.stack) > 0:
            return self.stack[-1]
        return None

    def __contains__(self, x):
        return x in self.stack

    def __len__(self):
        return len(self.stack)

    def __repr__(self):
        return self.stack.__str__()

    def __iter__(self):
        for x in self.stack:
            yield x

def taskMessage(t, task_to_programs):
    m = {
        "examples": [{"inputs": [xs[0].toJson()], "output": y.toJson()} for xs, y in t.examples],
        "name": t.name,
        "request": t.request.json(),
        "programs": task_to_programs[t.name]
    }
    return m

def execute_programs(tasks, grammar, task_to_programs_json):

    message = {
        "tasks": [taskMessage(t, task_to_programs_json) for t in tasks],
        "programTimeout": 0.1,
    }
    dumped_message = json.dumps(message)
    with open('message', 'w') as outfile:
        json.dump(message, outfile) 

    try:
        solver_file = os.path.join(get_root_dir(), "solvers/exec_arc_p")
        print("solver_file", solver_file)
        process = subprocess.Popen(
            solver_file, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        response, error = process.communicate(bytes(dumped_message, encoding="utf-8"))
        response = json.loads(response.decode("utf-8"))
        return response
        
    except OSError as exc:
        raise exc

def get_primitives_of_type(request, grammar):
	primitives = []
	for p in grammar.primitives:
		if p.tp.returns() == request:
			primitives.append(str(p))
	return primitives

def program_to_token_sequence(program, grammar):
	program_token_sequence = [token for token in program.left_order_tokens_alt()]
	return ["START"] + program_token_sequence


def main():

	path = "experimentOutputs/arc/2021-04-21T20:11:03.040547/"
	pickleFilename = "arc_arity=0_BO=False_CO=True_ES=1_ET=600_t_zero=3600_HR=0_it=2_MF=10_noConsolidation=True_RS=10000_RT=3600_RR=False_RW=False_solver=ocaml_STM=True_TRR=default_K=2_topkNotMAP=False_DSL=False.pickle"
	picklePath1 = path + pickleFilename
	result, _, _ = resume_from_path(picklePath1)

	for t,f in result.allFrontiers.items():
		if len(f.entries) > 0:
			print(f.topK(1).entries[0].program.size())

		for i,o in t.examples:
			print(i[0].toJson())
			print(o.toJson())
