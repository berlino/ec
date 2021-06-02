from dreamcoder.domains.list.listPrimitives import bootstrapTarget_extra
from dreamcoder.domains.list.taskProperties import handWrittenProperties, handWrittenPropertyFuncs, tinput, toutput
from dreamcoder.domains.list.makeListTasks import joshTasks
from dreamcoder.dreaming import backgroundHelmholtzEnumeration
from dreamcoder.enumeration import multicoreEnumeration
from dreamcoder.grammar import Grammar
from dreamcoder.likelihoodModel import PropertySignatureHeuristicModel, PropertyHeuristicModel, PropertySimTaskDiscriminatorHeuristic
from dreamcoder.program import *
from dreamcoder.recognition import variable
from dreamcoder.task import Task
from dreamcoder.type import tint, tlist


import copy
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class PropertySignatureExtractor(nn.Module):
    
    special = None
    
    def __init__(self, 
        tasks=[], 
        testingTasks=[], 
        cuda=False, 
        H=64,
        embedSize=16,
        # What should be the timeout for trying to construct Helmholtz tasks?
        helmholtzTimeout=0.25,
        # What should be the timeout for running a Helmholtz program?
        helmholtzEvaluationTimeout=0.01,
        grammar=None,
        featureExtractorArgs={},
        propertyType=arrow(tinput, toutput, tbool)
        ):
        super(PropertySignatureExtractor, self).__init__()

        print("Initializing PropertySignatureExtractor")

        self.special = "unique"
        self.CUDA = cuda
        self.recomputeTasks = True
        self.outputDimensionality = H
        self.useEmbeddings = not featureExtractorArgs["propNoEmbeddings"]
        self.featureExtractorArgs = featureExtractorArgs
        self.grammar = grammar

        self.tasks = tasks
        print("useEmbeddings: {}".format(self.useEmbeddings))

        self.propertyTasks = _convertToPropertyTasks(self.tasks, propertyType)
        print("Finished creating propertyTasks")

        if self.useEmbeddings:
            self.embedding = nn.Embedding(3, embedSize)
            self.embedSize = embedSize
        else:
            self.embedSize = 1

        # maps from a requesting type to all of the inputs that we ever saw with that request
        self.requestToInputs = {
            tp: [list(map(lambda ex: ex[0][0], t.examples)) for t in tasks if t.request == tp ]
            for tp in {t.request for t in tasks}
        }

        inputTypes = {t
                      for task in tasks
                      for t in task.request.functionArguments()}
        # maps from a type to all of the inputs that we ever saw having that type
        self.argumentsWithType = {
            tp: [ x
                  for t in tasks
                  for xs,_ in t.examples
                  for tpp, x in zip(t.request.functionArguments(), xs)
                  if tpp == tp]
            for tp in inputTypes
        }

        self.requestToNumberOfExamples = {
            tp: [ len(t.examples)
                  for t in tasks if t.request == tp ]
            for tp in {t.request for t in tasks}
        }
        self.helmholtzTimeout = helmholtzTimeout
        self.helmholtzEvaluationTimeout = helmholtzEvaluationTimeout
        self.parallelTaskOfProgram = True

        if cuda:
            self.CUDA=True
            self.cuda()  # I think this should work?

        self.properties = self._getProperties()

        self.linear = nn.Linear(len(self.properties) * self.embedSize, H)
        self.hidden = nn.Linear(H, H)
    
    def _getHelmholtzTasks(self, numHelmholtzTasks):
        """

        Returns:
            dreamTasks (list(Task)): python list of helmholtz-sampled Task objects
        """

        helmholtzFrontiers = backgroundHelmholtzEnumeration(tasks, dslGrammar, 3,
                                                            evaluationTimeout=0.001,
                                                            special="unique")
        frontiers = helmholtzFrontiers()
        random.shuffle(frontiers)
        programs = [frontier.entries[0].program for frontier in frontiers]

        dreamtTasks = []
        i = 0
        while len(dreamtTasks) < numHelmholtzTasks or i >= len(programs):
            task = self.taskOfProgram(programs[i], arrow(tlist(tint), tlist(tint)))
            if task is not None:
                dreamtTasks.append(task)
                print("program: {}".format(programs[i]))
                print("{} -> {}".format(task.examples[0][0], task.examples[0][1]))
            i += 1
        return dreamtTasks

    def _getPropertyGrammar(self):
        maxLL = max(list(self.grammar.expression2likelihood.values()))

        if self.featureExtractorArgs["propSamplingGrammar"] != "same":
            propertyPrimitives = self.featureExtractorArgs["primLibraries"][self.featureExtractorArgs["propSamplingGrammar"]]
        else:
            propertyPrimitives = self.grammar.primitives

        toutputToList = Primitive("toutput_to_tlist", arrow(toutput, tlist(tint)), lambda x: x)
        tinputToList = Primitive("tinput_to_tlist", arrow(tinput, tlist(tint)), lambda x: x)
        propertyPrimitives = propertyPrimitives + [toutputToList, tinputToList]

        if self.featureExtractorArgs["propAddZeroToNinePrims"]:

            for i in range(10):
                if str(i) not in [getattr(primitive, "name", "invented_primitive") for primitive in propertyPrimitives]:
                    propertyPrimitives.append(Primitive(str(i), tint, i))
        else:
            zeroToNinePrimitives = set([str(i) for i in range(10)])
            propertyPrimitives = [p for p in propertyPrimitives if p.name not in zeroToNinePrimitives]

        if self.featureExtractorArgs["propUseConjunction"]:
            propertyPrimitives.append(Primitive("and", arrow(tbool, tbool, tbool), lambda a: lambda b: a and b))

        productions = [(self.grammar.expression2likelihood.get(p, maxLL), p) for p in propertyPrimitives]
        propertyGrammar = Grammar.fromProductions(productions)
        print("property grammar: {}".format(propertyGrammar))
        return propertyGrammar


    def _getProperties(self):

        if self.featureExtractorArgs["propUseHandWrittenProperties"] is True:

            self.maxTaskInt = min(20, max([k for xs in self.argumentsWithType[tlist(tint)] for k in xs]))
            self.maxInputListLength = max([len(xs) for xs in self.argumentsWithType[tlist(tint)]])
            self.maxOutputListLength = self.maxInputListLength

            groupedProperties = handWrittenProperties(grouped=True)
            programs = handWrittenPropertyFuncs(groupedProperties, 0, self.maxTaskInt, self.maxInputListLength, self.maxOutputListLength)

        else:
            self.propertyGrammar = self._getPropertyGrammar()
            if self.featureExtractorArgs["propDreamTasks"]:
                dreamtTasks = sellf._getHelmholtzTasks(1000)
                properties = sampleProperties(self.featureExtractorArgs, self.propertyGrammar, dreamtTasks)
            else:
                frontierEntries = sampleProperties(self.featureExtractorArgs, self.propertyGrammar, self.propertyTasks)
            programs = [(None, frontierEntry.program) for frontierEntry in frontierEntries]
        
            print("Enumerated {} property programs".format(len(programs)))

        likelihoodModel = PropertySignatureHeuristicModel(tasks=self.propertyTasks)
        
        for programName, program in programs:

            print(programName)
            # print("p: {} (logprior: {})".format(frontierEntry.program, frontierEntry.logPrior))
            
            # the scoring is a function of all tasks which are already stored in likelihoodModel so
            # what we pass in the second argument does not matter

            needToEvaluate = False if self.featureExtractorArgs["propUseHandWrittenProperties"] else True
            programName = programName if programName is not None else str(program)
            likelihoodModel.score(program, self.propertyTasks[0], needToEvaluate, programName)

        print("{} out of {} properties after filtering".format(len(likelihoodModel.properties), len(programs)))
        # for program, f, propertyValues in likelihoodModel.properties:
        #     print(program)
        #     print(propertyValues)
        #     print('---------------------------------------------------')

        return likelihoodModel.properties[::]


    def forward(self, v, v2=None):

        v = torch.tanh(self.linear(v))
        v = torch.tanh(self.hidden(v))
        output = v.view(-1)
        return output

    def featuresOfTask(self, t):

        def getPropertyValue(propertyName, propertyFunc, t):
            """
            Args:
                propertyName (str): name of property
                propertyFunc (function):  property function of type (exampleInput -> exampleOutput -> {False, True, None})
                t (Task): task

            Returns:
                value_idx (int): The index of the property corresponding to propertyFunc for task t.
                0 corresponds to False, 1 corresponds to True and 2 corresponds to Mixed
            """

            taskPropertyValueToInt = {"allFalse":0, "allTrue":1, "mixed":0}

            specBooleanValues = []
            for example in t.examples:
                exampleInput, exampleOutput = example[0][0], example[1]
                try:
                    if not isinstance(exampleOutput, list):
                        exampleOutput = [exampleOutput]
                    if not isinstance(exampleInput, list):
                        exampleInput = [exampleInput]

                    booleanValue = propertyFunc(exampleInput)(exampleOutput)
                    # if propertyName == "output_idx_0_equals_input_idx_6":
                    #     print("{} -> {}".format(exampleInput, exampleOutput))
                    #     print(booleanValue)

                except Exception as e:
                    # if propertyName == "output_idx_0_equals_input_idx_6":
                    #     print("Failed to apply property: {}".format(propertyName))
                    #     print("{} -> {}".format(exampleInput, exampleOutput))
                    #     print(e)
                    #     print("------------------------------------------------")
                    booleanValue = None

                # property can't be applied to this io example and so property for the whole spec is Mixed (2)
                if booleanValue is None:
                    return taskPropertyValueToInt["mixed"]
                specBooleanValues.append(booleanValue)

            if all(specBooleanValues) is True:
                return taskPropertyValueToInt["allTrue"]

            elif all([booleanValue is False for booleanValue in specBooleanValues]):
                return taskPropertyValueToInt["allFalse"]
            return taskPropertyValueToInt["mixed"]

        propertyValues = []
        for propertyName, propertyProgram, _ in self.properties:
            propertyValue = getPropertyValue(propertyName, propertyProgram, t)
            propertyValues.append(propertyValue)
        
        booleanPropSig = torch.LongTensor(propertyValues)
        self.booleanPropSig = booleanPropSig

        # for i,el in enumerate(self.properties):
        #     print("{}: {}".format(el[0], booleanPropSig[i]))

        if self.useEmbeddings:
            embeddedPropSig = self.embedding(booleanPropSig).flatten()
            return self(embeddedPropSig)
        else:
            return self(booleanPropSig.float())


    def featuresOfTasks(self, ts, t2=None):  # Take a task and returns [features]
        """Takes the goal first; optionally also takes the current state second"""
        return [self.featuresOfTask(t) for t in ts]

    def taskOfProgram(self, p, tp):
        # half of the time we randomly mix together inputs
        # this gives better generalization on held out tasks
        # the other half of the time we train on sets of inputs in the training data
        # this gives better generalization on unsolved training tasks

        if random.random() < 0.5:
            def randomInput(t): return random.choice(self.argumentsWithType[t])
            # Loop over the inputs in a random order and pick the first ones that
            # doesn't generate an exception

            startTime = time.time()
            examples = []
            while True:
                # TIMEOUT! this must not be a very good program
                if time.time() - startTime > self.helmholtzTimeout: return None

                # Grab some random inputs
                xs = [randomInput(t) for t in tp.functionArguments()]
                try:
                    y = runWithTimeout(lambda: p.runWithArguments(xs), self.helmholtzEvaluationTimeout)
                    # print("Output y from below program: {}".format(y))
                    examples.append((tuple(xs),y))
                    # we want minimum 3 examples for each task
                    if len(examples) >= max(3, random.choice(self.requestToNumberOfExamples[tp])):
                        task = Task("Helmholtz", tp, examples)
                        return task
                except Exception as e:
                    # print(e)
                    # print("failed to apply program: {} \n to input: {}".format(p, xs))
                    continue

        else:
            candidateInputs = list(self.requestToInputs[tp])
            random.shuffle(candidateInputs)
            for xss in candidateInputs:
                ys = []
                for xs in xss:
                    try: y = runWithTimeout(lambda: p.runWithArguments(xs), self.helmholtzEvaluationTimeout)
                    except: break
                    ys.append(y)
                if len(ys) == len(xss):
                    return Task("Helmholtz", tp, list(zip(xss, ys)))
            return None


def testPropertySignatureFeatureExtractor(task_idx):
    from dreamcoder.domains.list.makeListTasks import make_list_bootstrap_tasks
    tasks = make_list_bootstrap_tasks()
    tasks = [task for task in tasks if task.request == arrow(tlist(tint), tlist(tint))]
    print("{} tasks".format(len(tasks)))
    task = tasks[task_idx]
    featureExtractor = PropertySignatureExtractor(tasks=tasks, useEmbeddings=False)

    print("Task: {}".format(task))
    for i,o in task.examples:
        print("{} -> {}".format(i[0], o))

    featureExtractor.featuresOfTask(task)
    propertySig = featureExtractor.test

    for i,propertyName in enumerate([propertyName for propertyName, propertyFunc in featureExtractor.properties]):
        print("{}: {}".format(propertyName, propertySig[i]))
    return


def sampleProperties(args, g, tasks, similarTasks=None, propertyRequest=arrow(tinput, toutput, tbool)):

    propertySamplingMethod = {
        "per_task_discrimination": PropertyHeuristicModel,
        "unique_task_signature": PropertySignatureHeuristicModel,
        "per_similar_task_discrimination": PropertySimTaskDiscriminatorHeuristic
    }[args["propScoringMethod"]]

    # if we sample properties by "unique_task_signature" we don't need to enumerate the same properties
    # for every task, as whether we choose to include the property or not depends on all tasks.

    propertyTasksToSolve = _convertToPropertyTasks(tasks, propertyRequest)
    propertySimilarTasks = _convertToPropertyTasks(similarTasks, propertyRequest)

    print("Enumerating with {} CPUs".format(args["propCPUs"]))
    frontiers, times, pcs, likelihoodModel = multicoreEnumeration(args["propSamplingGrammar"], propertyTasksToSolve,solver=args["propSolver"],maximumFrontier= int(10e7),
                                                 enumerationTimeout= args["propSamplingTimeout"], CPUs=args["propCPUs"],
                                                 evaluationTimeout=0.01,
                                                 testing=True, likelihoodModel=propertySamplingMethod, similarTasks=propertySimilarTasks)
    assert len(frontiers) == 1
    return frontiers[0].entries

def _convertToPropertyTasks(tasks, propertyRequest):
    propertyTasks = []
    for i,t in enumerate(tasks):
        print(i)
        # if i == 1981:
        #     continue
        tCopy = copy.deepcopy(t)
        tCopy.specialTask = ("property", None)
        tCopy.request = propertyRequest
        tCopy.examples = [(tuplify([io[0][0], io[1]]), True) for io in tCopy.examples]
        propertyTasks.append(tCopy)
    return propertyTasks


def testPropertySignatureExtractorHandwritten():

    def getTask(name, tasks):
        return [t for t in tasks if t.name == name][0]


    featureExtractorArgs = {
        "propCPUs": None,
        "propSolver": None,
        "propSamplingTimeout": None,
        "propUseConjunction": None,
        "propAddZeroToNinePrims": None,
        "propScoringMethod": None,
        "propDreamTasks": None,
        "propUseHandWrittenProperties": True,
        "propSamplingGrammar": None,
        "primLibraries": None
    }

    tasks = joshTasks("3")
    extractor = PropertySignatureExtractor(tasks=tasks, useEmbeddings=False, featureExtractorArgs=featureExtractorArgs)
    propertyNames = [el[0] for el in extractor.properties]
    
    task = getTask("005_1", tasks)
    extractor.featuresOfTask(task)
    v = extractor.test
    
    assert v[propertyNames.index("output_list_length_1")] == 1
    assert v[propertyNames.index("output_els_in_input")] == 1
    assert v[propertyNames.index("input_els_in_output")] == 2
    assert v[propertyNames.index("output_shorter_than_input")] == 1
    assert v[propertyNames.index("output_idx_0_equals_input_idx_3")] == 2

    task = getTask("003_1", tasks)
    extractor.featuresOfTask(task)
    v = extractor.test

    assert v[propertyNames.index("output_list_length_1")] == 1
    assert v[propertyNames.index("output_shorter_than_input")] == 1
    assert v[propertyNames.index("output_idx_0_equals_input_idx_6")] == 1
    assert v[propertyNames.index("output_idx_0_equals_input_idx_0")] == 2

    task = getTask("008_1", tasks)
    extractor.featuresOfTask(task)
    v = extractor.test

    assert v[propertyNames.index("output_list_length_1")] == 0
    assert v[propertyNames.index("output_list_length_6")] == 1
    assert v[propertyNames.index("output_shorter_than_input")] == 1
    assert v[propertyNames.index("output_idx_0_equals_input_idx_0")] == 1
    assert v[propertyNames.index("output_idx_1_equals_input_idx_1")] == 1
    assert v[propertyNames.index("output_idx_2_equals_input_idx_2")] == 1
    assert v[propertyNames.index("output_idx_3_equals_input_idx_3")] == 1
    assert v[propertyNames.index("output_idx_4_equals_input_idx_4")] == 1
    assert v[propertyNames.index("output_idx_5_equals_input_idx_5")] == 1
    assert v[propertyNames.index("output_idx_6_equals_input_idx_6")] == 2
    assert v[propertyNames.index("output_contains_input_idx_0")] == 1
    assert v[propertyNames.index("output_contains_input_idx_8")] == 2

    task = getTask("041_1", tasks)
    extractor.featuresOfTask(task)
    v = extractor.test

    assert v[propertyNames.index("output_contains_9")] == 1
    assert v[propertyNames.index("output_contains_8")] == 0
    assert v[propertyNames.index("all_output_els_mod_3_equals_0")] == 1
    assert v[propertyNames.index("all_output_els_mod_9_equals_0")] == 1
    assert v[propertyNames.index("all_output_els_mod_4_equals_0")] == 0
    assert v[propertyNames.index("all_output_els_lt_10")] == 1
    assert v[propertyNames.index("all_output_els_lt_3")] == 0


if __name__ == "__main__":
    pass
    




