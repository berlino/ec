from dreamcoder.program import *
from dreamcoder.recognition import variable
from dreamcoder.task import Task
from dreamcoder.domains.list.taskProperties import handWrittenProperties, handWrittenPropertyFuncs
from dreamcoder.type import tint, tlist

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
        helmholtzEvaluationTimeout=0.01):
        super(PropertySignatureExtractor, self).__init__()

        self.CUDA = cuda
        self.recomputeTasks = True
        self.outputDimensionality = H
        self.embedSize = embedSize

        if self.embedSize == 1:
            print("No embedding used for property values")
        else:
            self.embedding = nn.Embedding(3, self.embedSize)

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

        self.maxTaskInt = min(20, max([k for xs in self.argumentsWithType[tlist(tint)] for k in xs]))
        self.maxInputListLength = max([len(xs[0])
                  for t in tasks
                  for xs,_ in t.examples if t.request.arguments[0] == tlist(tint)])

        self.maxOutputListLength = max([len(y)
                  for t in tasks
                  for _,y in t.examples if t.request.returns() == tlist(tint)])

        print(self.maxTaskInt, self.maxInputListLength, self.maxOutputListLength)

        groupedProperties = handWrittenProperties()
        self.properties = [prop for subList in groupedProperties for prop in subList]
        self.propertyFuncs = handWrittenPropertyFuncs(groupedProperties, 0, self.maxTaskInt, self.maxInputListLength, self.maxOutputListLength)
        print("{} Properties used".format(len(self.propertyFuncs)))

        if cuda:
            self.CUDA=True
            self.cuda()  # I think this should work?

        self.linear = nn.Linear(len(self.propertyFuncs) * self.embedSize, H)
        self.hidden = nn.Linear(H, H)

    def forward(self, v, v2=None):

        v = F.tanh(self.linear(v))
        v = F.tanh(self.hidden(v))
        output = v.view(-1)
        return output

    def featuresOfTask(self, t):

        def getPropertyValue(propertyFunc, t):
            """
            Args:
                propertyFunc (function): property function of type (exampleInput -> exampleOutput -> {False, True, None})
                t (Task): task

            Returns:
                value_idx (int): The index of the property corresponding to propertyFunc for task t.
                0 corresponds to False, 1 corresponds to True and 2 corresponds to Mixed
            """
            mixed = 2 if self.embedSize > 1 else 0
            allTrue = 1
            allFalse = 0 if self.embedSize > 1 else -1

            specBooleanValues = []
            for example in t.examples[-1:]:
                exampleInput, exampleOutput = example[0][0], example[1]
                try:
                    booleanValue = propertyFunc(exampleOutput)(exampleInput)
                except Exception as e:
                    # print("Failed to apply property: {} to input {}".format(str(propertyFunc), exampleInput))
                    # print(e)
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
        for propertyFunc in self.propertyFuncs:
            propertyValue = getPropertyValue(propertyFunc, t)
            propertyValues.append(propertyValue)
        
        booleanPropSig = torch.LongTensor(propertyValues)
        if self.embedSize > 1:
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

if __name__ == "__main__":
    pass