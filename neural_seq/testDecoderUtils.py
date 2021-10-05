import pickle

from dreamcoder.domains.arc.arcPrimitives import basePrimitives, leafPrimitives, moreSpecificPrimitives

from neural_seq.decoderUtils import *

def test_parse_token_sequence():

    primitives = basePrimitives() + leafPrimitives() + moreSpecificPrimitives()
    grammar = Grammar.uniform(primitives)

    task_to_frontiers = pickle.load(open("data/arc/prior_enumeration_frontiers_8hr.pkl", "rb"))
    all_programs = [e.program for f in task_to_frontiers.values() for e in f.entries]

    for program in all_programs:
        token_sequence = program_to_token_sequence(program, grammar)
        print(token_sequence)
        print(program)
        print("-------------------------------------------------------------------------------------------------")
        # parsed_program = parse_token_sequence(token_sequence, grammar)
        # try:
        #     assert parsed_program == program
        # except AssertionError:
        #     print("\nGot: {}\nExpected: {}".format(parsed_program, program))
    return

def main():
    test_parse_token_sequence()


