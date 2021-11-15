import os

from dreamcoder.grammar import Grammar
from dreamcoder.domains.arc.arcPrimitives import basePrimitives, leafPrimitives, moreSpecificPrimitives, tgridin, tgridout
from dreamcoder.domains.arc.main import LMPseudoTranslationFeatureExtractor, retrieveARCJSONTasks
from dreamcoder.dreaming import helmholtzEnumeration
from dreamcoder.recognition import RecognitionModel
from dreamcoder.type import arrow

def main():

    # load tasks
    homeDirectory = "/".join(os.path.abspath(__file__).split("/")[:-4])
    dataDirectory = homeDirectory + "/arc_data/data/"
    tasks = retrieveARCJSONTasks(dataDirectory + 'training', useEvalExamplesForTraining=True, filenames=None)

    featureExtractor = LMPseudoTranslationFeatureExtractor([], testingTasks=[], cuda=False)

    request = arrow(tgridin, tgridout)
    grammar = Grammar.uniform(basePrimitives() + leafPrimitives() + moreSpecificPrimitives())

    recognitionModel = RecognitionModel(featureExtractor, grammar)
    frontiers = recognitionModel.sampleManyHelmholtz([request], N=1000, CPUs=20)

    # frontiers, _ = helmholtzEnumeration(grammar, request=request, inputs=None, timeout=1, special="arc")

    for f in frontiers:
        solution_programs = [e.program for e in f]
        f.task.solution_programs = solution_programs
        for e in f.entries:
            print("\nProgram:", e.program)
        print("Pseudotranslation:", featureExtractor.get_pseudo_translations(f.task)[0])


