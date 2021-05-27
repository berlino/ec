import torch
import torch.nn as nn
import torch.nn.functional as F
from dreamcoder.task import Task
from dreamcoder.recognition import variable

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class ArcCNN(nn.Module):
    special = 'arc'
    
    def __init__(self, tasks=[], testingTasks=[], cuda=False, H=64, inputDimensions=25):
        super(ArcCNN, self).__init__()

        self.CUDA = cuda
        self.recomputeTasks = True

        self.outputDimensionality = H
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                # nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

        self.gridDimension = 30

        # channels for hidden
        hid_dim = 64
        z_dim = 64

        self.encoder = nn.Sequential(
            conv_block(22, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
            Flatten()
        )

    def forward(self, v):
        """ """
        assert v.shape == (v.shape[0], 22, self.gridDimension, self.gridDimension)
        v = variable(v, cuda=self.CUDA).float()
        v = self.encoder(v)
        return v.mean(dim=0)


    def featuresOfTask(self, t):  # Take a task and returns [features]
        v = None
        if len(t.examples) < 1:
            # Return a zero torch variable.
            return variable(torch.zeros(self.outputDimensionality))
        for example in t.examples:
            inputGrid, outputGrid = example
            inputGrid = inputGrid[0]

            inputTensor = inputGrid.to_tensor(grid_height=30, grid_width=30)
            outputTensor = outputGrid.to_tensor(grid_height=30, grid_width=30)
            ioTensor = torch.cat([inputTensor, outputTensor], 0).unsqueeze(0)

            if v is None:
                v = ioTensor
            else:
                v = torch.cat([v, ioTensor], dim=0)
        return self(v)

    def taskOfProgram(self, p, tp):
        """
        For simplicitly we only use one example per task randomly sampled from
        all possible input grids we've seen.
        """
        def randomInput(t): return random.choice(self.argumentsWithType[t])

        startTime = time.time()
        examples = []
        while True:
            # TIMEOUT! this must not be a very good program
            if time.time() - startTime > self.helmholtzTimeout: return None

            # Grab some random inputs
            xs = [randomInput(t) for t in tp.functionArguments()]
            try:
                y = runWithTimeout(lambda: p.runWithArguments(xs), self.helmholtzEvaluationTimeout)
                examples.append((tuple(xs),y))
                if len(examples) >= 1:
                    return Task("Helmholtz", tp, examples)
            except: continue
        return None

    def featuresOfTasks(self, ts, t2=None):  # Take a task and returns [features]
        """Takes the goal first; optionally also takes the current state second"""
        return [self.featuresOfTask(t) for t in ts]
