from dreamcoder.domains.arc.utilsPostProcessing import *
from dreamcoder.program import Program
from dreamcoder.type import Context

def main():

	path = "experimentOutputs/arc/2021-04-21T20:11:03.040547/"
	pickleFilename = "arc_arity=0_BO=False_CO=True_ES=1_ET=600_t_zero=3600_HR=0_it=2_MF=10_noConsolidation=True_RS=10000_RT=3600_RR=False_RW=False_solver=ocaml_STM=True_TRR=default_K=2_topkNotMAP=False_DSL=False.pickle"
	picklePath1 = path + pickleFilename
	result, _, _ = resume_from_path(picklePath1)

	for t,f in result.allFrontiers.items():
		if len(f.entries) > 0:
			print(f.topK(1).entries[0].program.size())

def get_primitives_of_type(request, grammar):
	primitives = []
	for p in grammar.primitives:
		if p.tp.returns() == request:
			primitives.append(str(p))
	return primitives

def program_to_token_sequence(program, grammar):
	program_token_sequence = [token for token in program.left_order_tokens_alt()]
	return ["START"] + program_token_sequence


def parse_token_sequence(program_token_sequence, grammar):
	"""
	Args:
		program_primitive_sequence (list): A list of primtive indices
		grammar (Grammar): The grammar in which the program we want to parse is generated from

	Returns:
		program (Program): If the sequence can be parsed into a syntactically valid (i.e. satisfying type constraints)
		program then return that program, otherwise raise a parse error
	"""

	# types_queue = []

	# for primitive_idx in program_primitive_sequence:
	# 	primitive = grammar.primitives[primitive_idx]
	# 	# iterate through the arguments of the given primitive and record their types
	# 	for arg in primitive.tp.functionArguments():
	# 		types_queue.append(arg)

	# for token in program_token_sequence[1:]:

	return Program.parse("(lambda (to_min_grid (fill_color (merge_blocks (splitblocks_to_blocks (split_grid $0 false)) true) (nth_primary_color (grid_to_block $0) 0)) true))")

# program_primitive_sequence = [0, 4, 5, 12]