from dreamcoder.program import Primitive, Program
from dreamcoder.grammar import Grammar
from dreamcoder.type import tlist, tint, tbool, arrow, t0, t1, t2

def _isOutputSubsetOfInput(input_l, output_l):
	return set(output_l).issubset(set(input_l))

def _isOutputLengthOne(input_l, output_l):
	return len(output_l) == 1

def handWrittenProperties():
	return [
		Primitive("is_output_subset_of_input", arrow(tlist, tlist, tbool), _isOutputSubsetOfInput),
		Primitive("is_output_length_1", arrow(tlist, tlist, tbool), _isOutputLengthOne)
	]