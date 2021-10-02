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
    # frontiers = recognitionModel.sampleManyHelmholtz([request], N=100, CPUs=1)


    frontiers, _ = helmholtzEnumeration(grammar, requests=[request], enumerationTimeout=1, solver="ocaml", CPUs=1, maximumFrontier=1000)

    for f in frontiers:
        for e in f.entries:
            print(e.program)
