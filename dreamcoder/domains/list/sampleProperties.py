# experiments for Josh Rule
try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

import copy
import dill
import random

from dreamcoder.dreaming import *
from dreamcoder.domains.list.handwrittenProperties import handWrittenProperties, tinput, toutput
from dreamcoder.domains.list.main import retrieveJSONTasks
from dreamcoder.domains.list.makeListTasks import joshTasks, sortBootstrap
from dreamcoder.domains.list.listPrimitives import josh_primitives, bootstrapTarget_extra
from dreamcoder.domains.list.propertySignatureExtractor import PropertySignatureExtractor
from dreamcoder.domains.list.propSim import getPropertySimTasksMatrix, getPriorDistributionsOfProperties
from dreamcoder.domains.list.runUtils import list_options
from dreamcoder.domains.list.utilsPropertySampling import enumerateProperties
from dreamcoder.domains.list.utilsProperties import loadEnumeratedTasks
from dreamcoder.enumeration import *
from dreamcoder.grammar import Grammar
from dreamcoder.likelihoodModel import UniqueTaskSignatureScore
from dreamcoder.program import Program
from dreamcoder.type import *
from dreamcoder.utilities import tuplify

PRIMITIVES_TO_UPWEIGHT = [
    "toutput -> tlist(tint)", 
    "tinput -> tlist(tint)", 
    "index",
    "eq?",
    "length"
    ]

VALUES_TO_INT = {"allFalse":0, "allTrue":1, "mixed":2}

def prop_sampling_options(parser):
    parser.add_argument("--propSamplingTimeout",default=600,type=float)
    parser.add_argument("--propUseConjunction", action="store_true", default=False)
    parser.add_argument("--propAddZeroToNinePrims", action="store_true", default=False)
    parser.add_argument("--propEnumerationTimeout",default=1,type=float)
    parser.add_argument("--propSamplingMethod", default="unique_task_signature", choices=[
        "per_task_discrimination",
        "unique_task_signature"
        ])
    return parser

def upweight_primitives(primitives, primitivesToUpweight):
    expression2likelihood = {}
    for p in primitives:
        if p.name in primitivesToUpweight:
            expression2likelihood[p] = 4.0
    return expression2likelihood

def main(args):

    nameToPrimitives = {
        "josh_1": josh_primitives("1"),
        "josh_2": josh_primitives("2"),
        "josh_3": josh_primitives("3")[0],
        "josh_3.1": josh_primitives("3.1")[0],
        "josh_final": josh_primitives("final"),
        "property_prims": handWrittenProperties(),
        "dc_list_domain": bootstrapTarget_extra()
    }
    propertyPrimitives = nameToPrimitives[args["libraryName"]]
    if args["libraryName"] != "property_prims":
        toutputToList = Primitive("toutput_to_tlist", arrow(toutput, tlist(tint)), lambda x: x)
        tinputToList = Primitive("tinput_to_tlist", arrow(tinput, tlist(tint)), lambda x: x)
        propertyPrimitives = propertyPrimitives + [tinputToList, toutputToList]

    if args["propAddZeroToNinePrims"]:
        for i in range(10):
            if str(i) not in [primitive.name for primitive in propertyPrimitives]:
                propertyPrimitives.append(Primitive(str(i), tint, i))
    else:
        zeroToNinePrimitives = set([str(i) for i in range(10)])
        propertyPrimitives = [p for p in propertyPrimitives if p.name not in zeroToNinePrimitives]

    if args["propUseConjunction"]:
        propertyPrimitives.append(Primitive("and", arrow(tbool, tbool, tbool), lambda a: lambda b: a and b))

    tasks = {
        "Lucas-old": lambda: retrieveJSONTasks("data/list_tasks.json") + sortBootstrap(),
        "josh_1": lambda: joshTasks("1"),
        "josh_2": lambda: joshTasks("2"),
        "josh_3": lambda: joshTasks("3"),
        "josh_3.1": lambda: joshTasks("3.1"),
        "josh_final": lambda: joshTasks("final"),
    }[args["dataset"]]()
    

    if "josh" in args["dataset"]:
        tasks = [t for t in tasks if int(t.name[:3]) < 81 and "_1" in t.name]

    tasks = [t for t in tasks if (t.request == arrow(tlist(tint), tlist(tint)) and isinstance(t.examples[0][1],list) and isinstance(t.examples[0][0][0],list))]
    print("{} tasks".format(len(tasks)))

    expression2likelihood = {}
    # expression2likelihood = upweight_primitives(primitives, primitivesToUpweight)
    productions = [(expression2likelihood.get(p, 0.0), p) for p in propertyPrimitives]
    propertyGrammar = Grammar.fromProductions(productions)
    dslPrimitives = nameToPrimitives["josh_3"]
    grammar = Grammar.fromProductions([(0.0, p) for p in dslPrimitives])

    # instatiating the feature extractor samples properties if args["propToUse"] == "sample"
    featureExtractor = PropertySignatureExtractor(tasksToSolve=tasks, testingTasks=[], H=64, embedSize=16, helmholtzTimeout=0.001, helmholtzEvaluationTimeout=0.001,
            cuda=False, featureExtractorArgs=args, propertyGrammar=propertyGrammar,  grammar=grammar)

    helmholtzFrontiers = loadEnumeratedTasks(filename=args["helmholtzFrontiers"], primitives=dslPrimitives, numExamples=8)[:1000]
    propertySimTasksMatrix = getPropertySimTasksMatrix([f.task for f in helmholtzFrontiers], featureExtractor.properties, VALUES_TO_INT)
    propertyToPriorDistribution = getPriorDistributionsOfProperties(propertySimTasksMatrix, VALUES_TO_INT)
    for i,p in enumerate(featureExtractor.properties):
        p.setPropertyValuePriors(propertyToPriorDistribution[:, i], VALUES_TO_INT)

    for t in tasks:
        propertyScores = [(p, p.getPropertyValuePrior(p.getValue(t))) for p in featureExtractor.properties]
        sortedPropertyScores = sorted([(p,score) for p,score in propertyScores], key=lambda el: el[1])
        print("\n{}".format(t.describe()))
        for p,score in sortedPropertyScores[:5]:
            print(p.name, score, p.getValue(t))
