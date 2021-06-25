import dill
import torch

from dreamcoder.type import arrow, tlist, tint
from dreamcoder.recognition import RecognitionModel

def getGrammarsFromNeuralRecognizer(extractor, tasks, baseGrammar, featureExtractorArgs, sampledFrontiers, save, saveDirectory, args):

    recognitionModel = RecognitionModel(
    featureExtractor=extractor(tasks, grammar=baseGrammar, testingTasks=[], cuda=torch.cuda.is_available(), featureExtractorArgs=featureExtractorArgs),
    grammar=baseGrammar,
    cuda=torch.cuda.is_available(),
    contextual=False,
    previousRecognitionModel=False,
    )

    # count how many tasks can be tokenized
    excludeIdx = []
    for i,f in enumerate(sampledFrontiers):
        if recognitionModel.featureExtractor.featuresOfTask(f.task) is None:
            excludeIdx.append(i)
    sampledFrontiers = [f for i,f in enumerate(sampledFrontiers) if i not in excludeIdx]
    print("Can't get featuresOfTask for {} tasks. Now have {} frontiers".format(len(excludeIdx), len(sampledFrontiers)))

    ep, CPUs, helmholtzRatio, rs, rt = args.pop("earlyStopping"), args.pop("CPUs"), args.pop("helmholtzRatio"), args.pop("recognitionSteps"), args.pop("recognitionTimeout")
    
    # check if we alread have trained this model
    filename = "neural_ep={}_RS={}_RT={}_hidden={}_r={}_contextual={}.pkl".format(ep, rs, rt, featureExtractorArgs["hidden"], helmholtzRatio, args["contextual"])
    path = saveDirectory + filename
    try:
        grammars = open(path, 'rb')
        print("Loaded recognizer grammars from: {}".format(path))
        return grammars
    except FileNotFoundError:
        print("Trained recognizer not found, training now ...")


    trainedRecognizer = recognitionModel.trainRecognizer(
    frontiers=sampledFrontiers, 
    helmholtzFrontiers=[],
    helmholtzRatio=helmholtzRatio,
    CPUs=CPUs,
    lrModel=False, 
    earlyStopping=ep, 
    holdout=ep,
    steps=rs,
    timeout=rt,
    defaultRequest=arrow(tlist(tint), tlist(tint)))

    grammars = {}
    for task in tasks:
        grammar = trainedRecognizer.grammarOfTask(task).untorch()
        grammars[task] = grammar

    if save:
        with open(path, 'wb') as handle:
            print("Saved recognizer grammars at: {}".format(path))
            dill.dump(grammars, handle)

    return grammars

