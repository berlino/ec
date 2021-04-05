from dreamcoder.program import Primitive, Program
from dreamcoder.grammar import Grammar
from dreamcoder.type import tlist, tint, tbool, arrow, t0, t1, t2

def _everyOutputElGtEveryInputSameIdxEl(inputList, outputList):
	endIdx = min(len(inputList), len(outputList))
	return all([inputList[i] < outputList[i] for i in range(endIdx)])

def _aPrefixOfB(a, b):
	if len(a) > len(b):
		return False
	else:
		return all([a[i] == b[i] for i in range(len(a))])

def _aSuffixOfB(a, b):
	if len(a) > len(b):
		return False
	else:
		offset = len(b) - len(a)
		return all([a[i] == b[offset + i] for i in range(len(a))])

def handWrittenProperties():
	"""
	A list of properties to be used by PropertySignatureExtractor to create property signatures of tasks.
	These properties were hand written by solving the training tasks from the list domain official 
	experiments and thinking about what properties we could learn that make searching for the right
	programs easier for these tasks and hopefully also generalize to the test tasks.

	Ways to generate even more properties:
		2. Boolean Combination (usually AND e.g. land(input_subset_of_output, output_is_subset_of_input))
		3. Replace "all elements of list" with "any"

	Returns:
		properties: A list of Primitives which are the properties that we believe should be useful for search
	"""


	noParamProperties = [
		Primitive("output_subset_of_input", arrow(tlist(t0), tlist(t0), tbool), 
			lambda outputList: lambda inputList: set(outputList).issubset(set(inputList))),

		Primitive("input_subset_of_output", arrow(tlist(t0), tlist(t0), tbool), 
			lambda outputList: lambda inputList: set(inputList).issubset(set(outputList))),

		Primitive("output_same_length_as_input", arrow(tlist(t0), tlist(t1), tbool), 
			lambda outputList: lambda inputList: len(inputList) == len(outputList)),

		Primitive("output_shorter_than_input", arrow(tlist(t0), tlist(t1), tbool), 
			lambda outputList: lambda inputList: len(inputList) > len(outputList)),

		Primitive("output_list_longer_than_input", arrow(tlist(t0), tlist(t1), tbool), 
			lambda outputList: lambda inputList: len(inputList) < len(outputList)),

		Primitive("every_output_el_gt_every_input_same_idx_el", arrow(tlist(t0), tlist(t1), tbool), 
			lambda outputList: lambda inputList: _everyOutputElGtEveryInputSameIdxEl(inputList, outputList)),
 
		Primitive("input_prefix_of_output", arrow(tlist(t0), tlist(t0), tbool), 
			lambda outputList: lambda inputList: _aPrefixOfB(inputList ,outputList)),

		Primitive("input_suffix_of_output", arrow(tlist(t0), tlist(t0), tbool), 
			lambda outputList: lambda inputList: _aSuffixOfB(inputList, outputList)),

		Primitive("output_prefix_of_input", arrow(tlist(t0), tlist(t0), tbool), 
			lambda outputList: lambda inputList: _aPrefixOfB(outputList, inputList)),

		Primitive("output_suffix_of_input", arrow(tlist(t0), tlist(t0), tbool), 
			lambda outputList: lambda inputList: _aSuffixOfB(outputList, inputList))
		]

	kParamProperties = [
		Primitive("all_output_els_mod_k_equals_0", arrow(tlist(t0), tlist(tint), tint, tbool), 
			lambda k: lambda outputList: lambda inputList: all([el % k == 0 for el in outputList]) if k > 0 else None),
		
		Primitive("all_output_els_lt_k", arrow(tlist(t0), tlist(tint), tint, tbool), 
			lambda k: lambda outputList: lambda inputList: all([el < k for el in outputList])),

		Primitive("output_contains_k", arrow(tlist(t0), tlist(t1), t1, tbool),
			lambda k: lambda outputList: lambda inputList: (k in outputList)),
	]


	inputIdxParamProperties = [
		Primitive("output_contains_input_idx_i", arrow(tlist(t0), tlist(t1), tint, tbool),
			lambda i: lambda outputList: lambda inputList: None if i >= len(inputList) else inputList[i] in outputList),
	]

	outputIdxParamProperties = [
		Primitive("output_list_length_n", arrow(tlist(t0), tlist(t1), tint, tbool), 
			lambda n: lambda outputList: lambda inputList: len(outputList) == n)
	]

	inputIdxOutputIdxParamProperties = [
		# Primitive("output_idx_i_equals_input_idx_j", arrow(tlist(tint), tlist(tint), tint, tint, tbool),
		# 	lambda i: lambda j: lambda outputList: lambda inputList: None if (j >= len(inputList) or i >= len(outputList)) else inputList[j] == outputList[i])
	]

	return [noParamProperties, 
			kParamProperties, 
			inputIdxParamProperties, 
			outputIdxParamProperties, 
			inputIdxOutputIdxParamProperties]


def handWrittenPropertyFuncs(handWrittenPropertyPrimitives, kMin, kMax, 
	inputIdxMax, outputIdxMax):

	propertyFuncs = []

	noParamProperties = handWrittenPropertyPrimitives[0]
	for prop in noParamProperties:
		propertyFuncs.append((prop.name, prop.value))

	kParamProperties = handWrittenPropertyPrimitives[1]
	for prop in kParamProperties:
		for k in range(kMin, kMax+1):
			propertyFuncs.append(("{}_k={}".format(prop.name, k), prop.value(k)))

	inputIdxParamProperties = handWrittenPropertyPrimitives[2]
	for prop in inputIdxParamProperties:
		for i in range(inputIdxMax+1):
			propertyFuncs.append(("{}_i={}".format(prop.name, i), prop.value(i)))

	outputIdxParamProperties = handWrittenPropertyPrimitives[3]
	for prop in outputIdxParamProperties:
		for n in range(outputIdxMax+1):
			propertyFuncs.append((("{}_n={}".format(prop.name, n), prop.value(n))))

	inputIdxOutputIdxParamProperties = handWrittenPropertyPrimitives[4]
	for prop in inputIdxOutputIdxParamProperties:
		for j in range(inputIdxMax+1):
			for i in range(outputIdxMax+1):
				propertyFuncs.append(("{}_j={}_i={}".format(prop.name, j, i),prop.value(i)(j)))

	return propertyFuncs





