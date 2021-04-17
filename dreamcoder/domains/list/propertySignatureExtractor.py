from dreamcoder.domains.list.listPrimitives import bootstrapTarget_extra
from dreamcoder.domains.list.taskProperties import handWrittenProperties, handWrittenPropertyFuncs, tinput, toutput
from dreamcoder.dreaming import backgroundHelmholtzEnumeration
from dreamcoder.enumeration import multicoreEnumeration
from dreamcoder.grammar import Grammar
from dreamcoder.likelihoodModel import PropertySignatureHeuristicModel, PropertyHeuristicModel
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
        useEmbeddings=True, 
        embedSize=16,
        # What should be the timeout for trying to construct Helmholtz tasks?
        helmholtzTimeout=0.25,
        # What should be the timeout for running a Helmholtz program?
        helmholtzEvaluationTimeout=0.01,
        grammar=None,
        featureExtractorArgs={}
        ):
        super(PropertySignatureExtractor, self).__init__()

        print("Initializing PropertySignatureExtractor")

        self.special = "unique"
        self.CUDA = cuda
        self.recomputeTasks = True
        self.outputDimensionality = H
        self.useEmbeddings = useEmbeddings
        self.featureExtractorArgs = featureExtractorArgs
        self.grammar = grammar

        self.tasks = tasks
        print()

        request = arrow(tinput, toutput, tbool)
        self.propertyTasks = []
        for t in self.tasks:
            tCopy = copy.deepcopy(t)
            tCopy.specialTask = ("property", None)
            tCopy.request = request
            tCopy.examples = [(tuplify([io[0][0], io[1]]), True) for io in tCopy.examples]
            self.propertyTasks.append(tCopy)


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

        for propName, propFunc, propSig in self.properties:
            print("{}: {}".format(propName, propFunc))
            if propSig is not None:
                print("Tasks Sig: {}".format(propSig))

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

        self.propertyGrammar = self._getPropertyGrammar()
        if self.featureExtractorArgs["propUseHandWrittenProperties"] is True:

            print(self.argumentsWithType.keys())

            self.maxTaskInt = min(20, max([k for xs in self.argumentsWithType[tlist(tint)] for k in xs]))
            self.maxInputListLength = max([len(xs) for xs in self.argumentsWithType[tlist(tint)]])
            self.maxOutputListLength = self.maxInputListLength

            groupedProperties = handWrittenProperties(grouped=True)
            properties = handWrittenPropertyFuncs(groupedProperties, 0, self.maxTaskInt, self.maxInputListLength, self.maxOutputListLength)
            return properties

        else:
            if self.featureExtractorArgs["propDreamTasks"]:
                dreamtTasks = sellf._getHelmholtzTasks(1000)
                properties = sampleProperties(self.featureExtractorArgs, self.propertyGrammar, dreamtTasks)
            else:
                frontierEntries = sampleProperties(self.featureExtractorArgs, self.propertyGrammar, self.propertyTasks)
            programs = [frontierEntry.program for frontierEntry in frontierEntries]
        
            print("Enumerated {} property programs".format(len(programs)))

            likelihoodModel = PropertySignatureHeuristicModel(tasks=self.propertyTasks)
            for program in programs:
                # print("p: {} (logprior: {})".format(frontierEntry.program, frontierEntry.logPrior))
                
                # the scoring is a function of all tasks which are already stored in likelihoodModel so
                # what we pass in the second argument does not matter
                _, score = likelihoodModel.score(program, self.propertyTasks[0])
            print("{} properties after filtering".format(len(likelihoodModel.properties)))
            for program, propertyValues in likelihoodModel.properties:
                print(program)
                print(propertyValues)
                print('---------------------------------------------------')

            return [(str(program), program.evaluate([]), propertyValues) for program, propertyValues in likelihoodModel.properties]


    def forward(self, v, v2=None):

        v = F.tanh(self.linear(v))
        v = F.tanh(self.hidden(v))
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

            mixed = 2 if self.embedSize > 1 else 0
            allTrue = 1
            allFalse = 0 if self.embedSize > 1 else -1

            specBooleanValues = []
            for example in t.examples:
                exampleInput, exampleOutput = example[0][0], example[1]
                try:
                    if not isinstance(exampleOutput, list):
                        exampleOutput = [exampleOutput]
                    if not isinstance(exampleInput, list):
                        exampleInput = [exampleInput]
                    booleanValue = propertyFunc(exampleOutput)(exampleInput)
                except Exception as e:
                    # print("Failed to apply property: {}".format(propertyName))
                    # print("{} -> {}".format(exampleInput, exampleOutput))
                    # print(e)
                    # print("------------------------------------------------")
                    booleanValue = None

                # property can't be applied to this io example and so property for the whole spec is Mixed (2)
                if booleanValue is None:
                    return mixed
                specBooleanValues.append(booleanValue)

            if all(specBooleanValues) is True:
                return allTrue

            elif all([booleanValue is False for booleanValue in specBooleanValues]):
                return allFalse
            return mixed

        propertyValues = []
        for propertyName, propertyProgram, _ in self.properties:
            propertyValue = getPropertyValue(propertyName, propertyProgram, t)
            propertyValues.append(propertyValue)
        
        booleanPropSig = torch.LongTensor(propertyValues)
        self.test = booleanPropSig

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
                    if len(examples) >= random.choice(self.requestToNumberOfExamples[tp]):
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


def sampleProperties(args, g, tasks):

    propertySamplingMethod = {
        "per_task_discrimination": PropertyHeuristicModel,
        "unique_task_signature": PropertySignatureHeuristicModel
    }[args["propSamplingMethod"]]

    # if we sample properties in this way, we don't need to enumerate the same properties
    # for every task, as whether we choose to include the property or not depends on all tasks.
    if args["propSamplingMethod"] == "unique_task_signature":
        tasksForPropertySampling = [tasks[0]]
    else:
        tasksForPropertySampling = tasks

    print("Enumerating with {} CPUs".format(args["propCPUs"]))
    frontiers, times, pcs, likelihoodModel = multicoreEnumeration(g,tasksForPropertySampling,solver=args["propSolver"],maximumFrontier= int(10e7),
                                                 enumerationTimeout= args["propSamplingTimeout"], CPUs=args["propCPUs"],
                                                 evaluationTimeout=0.01,
                                                 testing=True, allTasks=tasks, likelihoodModel=propertySamplingMethod)

    if args["propSamplingMethod"] == "unique_task_signature":
        assert len(frontiers) == 1
        return frontiers[0].entries
    elif args["propSamplingMethod"] == "per_task_discrimination":
        raise Exception("Not implemented yet")
    return None


if __name__ == "__main__":
    pass




