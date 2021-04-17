from dreamcoder.program import Primitive, Program
from dreamcoder.grammar import Grammar
from dreamcoder.type import tlist, tint, tbool, arrow, baseType, t1

def _everyOutputElGtEveryInputSameIdxEl(inputList, outputList):
	endIdx = min(len(inputList), len(outputList))
	return all([inputList[i] < outputList[i] for i in range(endIdx)])

def _inputPrefixOfOutput(inputList, outputList):
	if len(inputList) > len(outputList):
		return False
	else:
		return all([inputList[i] == outputList[i] for i in range(len(inputList))])

def _inputSuffixOfOutput(inputList, outputList):
	if len(inputList) > len(outputList):
		return False
	else:
		offset = len(outputList) - len(inputList)
		return all([inputList[i] == outputList[offset + i] for i in range(len(inputList))])

tinput = baseType("input")
toutput = baseType("output")

def handWrittenProperties(grouped=False):
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
		Primitive("output_subset_of_input", arrow(tinput, toutput, tbool), 
			lambda inputList: lambda outputList: set(outputList).issubset(set(inputList))),

		Primitive("input_subset_of_output", arrow(tinput, toutput, tbool), 
			lambda inputList: lambda outputList: set(inputList).issubset(set(outputList))),

		Primitive("output_same_length_as_input", arrow(tinput, toutput, tbool), 
			lambda inputList: lambda outputList: len(inputList) == len(outputList)),

		Primitive("output_shorter_than_input", arrow(tinput, toutput, tbool), 
			lambda inputList: lambda outputList: len(inputList) > len(outputList)),

		Primitive("output_list_longer_than_input", arrow(tinput, toutput, tbool), 
			lambda inputList: lambda outputList: len(inputList) < len(outputList)),

		Primitive("every_output_el_gt_every_input_same_idx_el", arrow(tinput, toutput, tbool), 
			lambda inputList: lambda outputList: _everyOutputElGtEveryInputSameIdxEl(inputList, outputList)),
 
		Primitive("input_prefix_of_output", arrow(tinput, toutput, tbool), 
			lambda inputList: lambda outputList: _inputPrefixOfOutput(inputList, outputList)),

		Primitive("input_suffix_of_output", arrow(tinput, toutput, tbool), 
			lambda inputList: lambda outputList: _inputSuffixOfOutput(inputList, outputList))
		]

	kParamProperties = [
		Primitive("all_output_els_mod_k_equals_0", arrow(tinput, toutput, tint, tbool), 
			lambda inputList: lambda outputList: lambda k: all([el % k == 0 for el in outputList]) if k > 0 else None),
		
		Primitive("all_output_els_lt_k", arrow(tinput, toutput, tint, tbool), 
			 lambda inputList: lambda outputList: lambda k: all([el < k for el in outputList])),

		Primitive("output_contains_k", arrow(tinput, toutput, tint, tbool),
			 lambda inputList: lambda outputList: lambda k: k in outputList),
	]

	inputIdxParamProperties = [
		Primitive("output_contains_input_idx_i", arrow(tinput, toutput, tint, tbool),
			lambda inputList: lambda outputList: lambda i: None if i >= len(inputList) else inputList[i] in outputList),
	]

	outputIdxParamProperties = [
		Primitive("output_list_length_n", arrow(tinput, toutput, tint, tbool), 
			lambda inputList: lambda outputList: lambda n: len(outputList) == n)
	]

	inputIdxOutputIdxParamProperties = [
		Primitive("output_idx_i_equals_input_idx_j", arrow(tinput, toutput, tint, tint, tbool),
			lambda inputList: lambda outputList: lambda j: lambda i: None if (j >= len(inputList) or i >= len(outputList)) else inputList[j] == outputList[i])
	]

	if grouped:
		return [noParamProperties, kParamProperties, inputIdxParamProperties, outputIdxParamProperties, inputIdxOutputIdxParamProperties]
	else:
		return noParamProperties + kParamProperties + inputIdxParamProperties + outputIdxParamProperties + inputIdxOutputIdxParamProperties


def handWrittenPropertyFuncs(handWrittenPropertyPrimitives, kMin, kMax, 
	inputIdxMax, outputIdxMax):

	propertyFuncs = []

	noParamProperties = handWrittenPropertyPrimitives[0]
	for prop in noParamProperties:
		propertyFuncs.append((prop.name, prop.value, None))

	kParamProperties = handWrittenPropertyPrimitives[1]
	for prop in kParamProperties:
		for k in range(kMin, kMax+1):
			propertyFuncs.append((prop.name.replace("_k", "_{}".format(k)), prop.value(k), None))

	inputIdxParamProperties = handWrittenPropertyPrimitives[2]
	for prop in inputIdxParamProperties:
		for idx in range(inputIdxMax+1):
			propertyFuncs.append((prop.name.replace("idx_i", "idx_{}".format(idx)), prop.value(idx), None))

	outputIdxParamProperties = handWrittenPropertyPrimitives[3]
	for prop in outputIdxParamProperties:
		for idx in range(outputIdxMax+1):
			propertyFuncs.append((prop.name.replace("_n", "_{}".format(idx)), prop.value(idx), None))

	inputIdxOutputIdxParamProperties = handWrittenPropertyPrimitives[4]
	for prop in inputIdxOutputIdxParamProperties:
		for inputIdx in range(inputIdxMax+1):
			for outputIdx in range(outputIdxMax+1):
				propertyFuncs.append((
					prop.name.replace("idx_j", "idx_{}".format(inputIdx)).replace("idx_i", "idx_{}".format(outputIdx)), 
					prop.value(outputIdx)(inputIdx), 
					None))

	return propertyFuncs





