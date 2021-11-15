import os

from dreamcoder.domains.arc.arcPrimitives import basePrimitives, leafPrimitives, moreSpecificPrimitives
from dreamcoder.domains.arc.experiment_output import ocaml_execute_programs, load_relevant_data
from dreamcoder.program import Program, ParseFailure, InferenceFailure, ShiftFailure, RunFailure, EtaExpandFailure
from dreamcoder.fragmentUtilities import MatchFailure

def loads_task_to_programs(path):
	"""
	Returns:
		task_to_programs (dict): A dictionary of with task_name (str) keys and list of program tuple values. The
		first element of each program tuple is the program string and the second element is its log prior (or None)
	"""
	task_to_programs = {}
	for filename in os.listdir(path):
		task_to_programs[filename] = []
		with open(path + "/" + filename) as f:
			for line in f.readlines():
				# Every line starts with 'Program: ' which we want to exclude from program_string
				program_string = line.strip()[line.index(": ")+2:]
				task_to_programs[filename].append((program_string, None))
	return task_to_programs

def filter_syntactically_invalid_programs(task_to_programs):
	syntactically_valid_task_to_programs = {}
	for t, programs in task_to_programs.items():
		syntactically_valid_task_to_programs[t] = []
		for p_string,_ in programs:
			try:
				p = Program.parse(p_string)
				synt_valid_task_to_programs[t].append((p_string, None))
			except ParseFailure as e:
				print("ParseFailure", e)
				pass
			except InferenceFailure as e:
				print("InferenceFailure", e)
			except ShiftFailure as e:
				print("ShiftFailure", e)
			except RunFailure as e:
				print("RunFailure", e)
			except EtaExpandFailure as e:
				print("EtaExpandFailure", e)
			except MatchFailure as e:
				print("MatchFailure", e)
			except Exception as e:
				print("OtherFailure", e)
	return syntactically_valid_task_to_programs

def codex_synthesis_main():

	# Program.parse looks for Primitives in global scope
	basePrimitives()
	leafPrimitives()
	moreSpecificPrimitives()

	path = "data/larc/codex_predictions"
	task_to_programs = loads_task_to_programs(path)
	task_to_programs = filter_syntactically_invalid_programs(task_to_programs)

	_, _, _, _, grammar, tasks, _ = load_relevant_data()
	ocaml_execute_programs(grammar, tasks, task_to_programs)

	return 
	