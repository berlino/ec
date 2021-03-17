from dreamcoder.recognition import variable
from dreamcoder.domains.list.taskProperties import handWrittenProperties

import torch.nn as nn
import torch.nn.functional as F


class PropertySignatureExtractor(nn.Module):
    special = 'prop_sig'
    
    def __init__(self, tasks=[], testingTasks=[], cuda=False, H=64, embed_size=16, properties=handWrittenProperties()):
        super(PropertySignatureExtractor, self).__init__()
        self.CUDA = cuda
        self.recomputeTasks = True
        self.outputDimensionality = H
        self.embed_size = embed_size
        self.properties = properties

        if cuda:
            self.CUDA=True
            self.cuda()  # I think this should work?

        self.linear = nn.Linear(len(self.properties) * self.embed_size, H)
        # self.hidden = nn.Linear(H, H)

    def forward(self, v, v2=None):

        v = F.relu(self.linear(v))
        return v.view(-1)

    def featuresOfTask(self, t):

        for property_primitive in self.properties:
            property_func = property_primitive.value
            for example in t.examples[-1:]:
                spec_input, spec_output = example
                spec_input = spec_input[0]
        return self(v)

    def featuresOfTasks(self, ts, t2=None):  # Take a task and returns [features]
        """Takes the goal first; optionally also takes the current state second"""
        return [self.featuresOfTask(t) for t in ts]

if __name__ == "__main__":
    pass