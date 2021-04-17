# experiments for Josh Rule
try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

import copy
import random

from dreamcoder.dreaming import *
from dreamcoder.domains.list.main import retrieveJSONTasks
from dreamcoder.domains.list.makeListTasks import joshTasks, sortBootstrap
from dreamcoder.domains.list.listPrimitives import josh_primitives, bootstrapTarget_extra
from dreamcoder.domains.list.propertySignatureExtractor import PropertySignatureExtractor, sampleProperties
from dreamcoder.domains.list.taskProperties import handWrittenProperties, tinput, toutput
from dreamcoder.enumeration import *
from dreamcoder.grammar import Grammar
from dreamcoder.likelihoodModel import PropertySignatureHeuristicModel, PropertyHeuristicModel
from dreamcoder.program import Program
from dreamcoder.type import *
from dreamcoder.utilities import tuplify


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "")
    # parser.add_argument("--dream","-d",default=False,action='store_true')
    parser.add_argument("--samplingTimeout",default=600,type=float)
    parser.add_argument("--CPUs",default=numberOfCPUs(),type=int)
    parser.add_argument("--solver",default="ocaml",type=str)
    parser.add_argument("--primitives", default="property_prims", choices=[
        "josh_1",
        "josh_2",
        "josh_3",
        "josh_3.1",
        "josh_final",
        "property_prims",
        "list_prims"])
    parser.add_argument(
        "--dataset",
        type=str,
        default="Lucas-old",
        choices=[
            "josh_1",
            "josh_2",
            "josh_3",
            "josh_3.1",
            "josh_final",
            "Lucas-old"])
    parser.add_argument("--propUseConjunction", action="store_true", default=False)
    parser.add_argument("--propAddZeroToNinePrims", action="store_true", default=False)
    parser.add_argument("--propertySamplingMethod", default="unique_task_signature", choices=[
        "per_task_discrimination",
        "unique_task_signature"
        ])
    parser.add_argument("--dreamTasks", action="store_true", default=False)
    parser.add_argument("--useHandWrittenProperties", action="store_false", default=True)

    args = parser.parse_args()

    nameToPrimitives = {
        "josh_1": josh_primitives("1"),
        "josh_2": josh_primitives("2"),
        "josh_3": josh_primitives("3")[0],
        "josh_3.1": josh_primitives("3.1")[0],
        "josh_final": josh_primitives("final"),
        "property_prims": handWrittenProperties(),
        "list_prims": bootstrapTarget_extra()
    }
    propertyPrimitives = nameToPrimitives[args.primitives]

    if args.primitives != "property_prims":
        toutputToList = Primitive("toutput_to_tlist", arrow(toutput, tlist(tint)), lambda x: x)
        tinputToList = Primitive("tinput_to_tlist", arrow(tinput, tlist(tint)), lambda x: x)
        propertyPrimitives = propertyPrimitives + [tinputToList, toutputToList]

    if args.propAddZeroToNinePrims:
        for i in range(10):
            if str(i) not in [primitive.name for primitive in propertyPrimitives]:
                propertyPrimitives.append(Primitive(str(i), tint, i))
    else:
        zeroToNinePrimitives = set([str(i) for i in range(10)])
        propertyPrimitives = [p for p in propertyPrimitives if p.name not in zeroToNinePrimitives]

    if args.propUseConjunction:
        propertyPrimitives.append(Primitive("and", arrow(tbool, tbool, tbool), lambda a: lambda b: a and b))

    tasks = {
        "Lucas-old": lambda: retrieveJSONTasks("data/list_tasks.json") + sortBootstrap(),
        "josh_1": lambda: joshTasks("1"),
        "josh_2": lambda: joshTasks("2"),
        "josh_3": lambda: joshTasks("3"),
        "josh_3.1": lambda: joshTasks("3.1"),
        "josh_final": lambda: joshTasks("final"),
    }[args.dataset]()
    

    if "josh" in args.dataset:
        tasks = [t for t in tasks if int(t.name[:3]) < 81 and "_1" in t.name]

    tasks = [t for t in tasks if (t.request == arrow(tlist(tint), tlist(tint)) and isinstance(t.examples[0][1],list) and isinstance(t.examples[0][0][0],list))]
    print("{} tasks".format(len(tasks)))

    expression2likelihood = {}
    # primitivesToUpweight = [
    # "toutput -> tlist(tint)", 
    # "tinput -> tlist(tint)", 
    # "index",
    # "eq?",
    # "length"
    # ]
    # for p in primitives:
    #     if p.name in primitivesToUpweight:
    #         expression2likelihood[p] = 4.0
    productions = [(expression2likelihood.get(p, 0.0), p) for p in propertyPrimitives]
    propertyGrammar = Grammar.fromProductions(productions)

    dslPrimitives = nameToPrimitives["josh_3"]
    dslGrammar = Grammar.fromProductions([(0.0, p) for p in dslPrimitives])

    def scoreProgram(p, request, recognizer=None, grammar=None):

        if recognizer is not None:
            grammar = recognizer.grammarOfTask(task).untorch()

        ll = grammar.logLikelihood(request, p)
        return ll

    properties = [
        "(lambda (lambda (eq? (car (tinput_to_tlist $1)) (car (toutput_to_tlist $0)))))",
        "(lambda (lambda (eq? (fix1 (toutput_to_tlist $0) (lambda (lambda (if (empty? $0) 0 (+n9 ($1 (cdr $0)) 1))))) 1)))"

    ]

    print(propertyGrammar)

    request = arrow(tinput, toutput, tbool)
    for p in properties:
        ll = scoreProgram(Program.parse(p), request, grammar=propertyGrammar)
        print(ll)

    # print(dslGrammar)
    # featureExtractor = PropertySignatureExtractor(tasks=tasks, testingTasks=[], H=64, embedSize=16, useEmbeddings=True, helmholtzTimeout=0.001, helmholtzEvaluationTimeout=0.001,
    #         cuda=False, doSampling=True, args=args, propertyGrammar=propertyGrammar,  dslGrammar=dslGrammar)


    # frontierEntries = sampleProperties(args, propertyGrammar, tasks)
    # print("Enumerated {} properties".format(len(frontierEntries)))

    # likelihoodModel = PropertySignatureHeuristicModel(tasks=tasks)
    # for frontierEntry in frontierEntries:
    #     # print("p: {} (logprior: {})".format(frontierEntry.program, frontierEntry.logPrior))
    #     _, score = likelihoodModel.score(frontierEntry.program, tasks[0])

    # print("{} properties after filtering".format(len(likelihoodModel.properties)))
    # for prop, propertyValues in likelihoodModel.properties:
    #     print(prop)
    #     print(propertyValues)


