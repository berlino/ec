from dreamcoder.recognition import variable
from dreamcoder.domains.list.taskProperties import handWrittenProperties, handWrittenPropertyFuncs

import torch
import torch.nn as nn
import torch.nn.functional as F


class PropertySignatureExtractor(nn.Module):
    
    special = 'prop_sig'
    
    def __init__(self, tasks=[], testingTasks=[], cuda=False, H=64, embedSize=16):
        super(PropertySignatureExtractor, self).__init__()
        self.CUDA = cuda
        self.recomputeTasks = True
        self.outputDimensionality = H
        self.embedSize = embedSize
        self.embedding = nn.Embedding(3, self.embedSize)

        groupedProperties = handWrittenProperties()
        self.properties = [prop for subList in groupedProperties for prop in subList]
        self.propertyFuncs = handWrittenPropertyFuncs(groupedProperties, 0, 10, 10, 10)

        if cuda:
            self.CUDA=True
            self.cuda()  # I think this should work?

        self.linear = nn.Linear(len(self.propertyFuncs) * self.embedSize, H)
        # self.hidden = nn.Linear(H, H)

    def forward(self, v, v2=None):

        v = F.relu(self.linear(v))
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
            specBooleanValues = []
            for example in t.examples[-1:]:
                exampleInput, exampleOutput = example[0][0], example[1]
                try:
                    booleanValue = propertyFunc(exampleOutput)(exampleInput)
                except Exception as e:
                    print(e)
                    booleanValue = None

                # property can't be applied to this io example and so property for the whole spec is Mixed (2)
                if booleanValue is None:
                    return 2
                specBooleanValues.append(booleanValue)

            if all(specBooleanValues) is True:
                return 1
            elif all([booleanValue is False for booleanValue in specBooleanValues]):
                return 0
            return 2

        propertyValues = []
        for propertyFunc in self.propertyFuncs:
            propertyValue = getPropertyValue(propertyFunc, t)
            propertyValues.append(propertyValue)
        
        booleanPropSig = torch.LongTensor(propertyValues)
        embeddedPropSig = self.embedding(booleanPropSig).flatten()

        return self(embeddedPropSig)

    def featuresOfTasks(self, ts, t2=None):  # Take a task and returns [features]
        """Takes the goal first; optionally also takes the current state second"""
        return [self.featuresOfTask(t) for t in ts]

if __name__ == "__main__":
    pass