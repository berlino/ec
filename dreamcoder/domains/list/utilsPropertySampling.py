import dill
import random

from dreamcoder.domains.list.utilsProperties import convertToPropertyTasks
from dreamcoder.likelihoodModel import TaskDiscriminationScore


def getPropertySamplingGrammar(baseGrammar, grammarName, frontiers, pseudoCounts=1, seed=0):
    if grammarName == "random":
        random.seed(seed)
        grammar = baseGrammar.randomWeights(r=lambda oldWeight: -1 * random.uniform(0,5))
    elif grammarName == "fitted":
        minLogPrior = -11
        frontiersToFitOn = [f for f in frontiers if (len(f.entries) > 0 and baseGrammar.logLikelihood(f.task.request, f.topK(1).entries[0].program) > minLogPrior)]
        print("Fitting on {} frontiers with logPrior > {}".format(len(frontiersToFitOn), minLogPrior))
        grammar = baseGrammar.insideOutside(frontiersToFitOn, pseudoCounts=pseudoCounts)
    elif grammarName == "base":
        grammar = baseGrammar
    return grammar


def sampleProperties(grammar, tasks, propertyRequest, featureExtractor, featureExtractorArgs, save):
    try:
        propertyFeatureExtractor = featureExtractor(tasks=tasks, similarTasks=None, grammar=grammar, testingTasks=[], cuda=False, featureExtractorArgs=featureExtractorArgs, propertyRequest=propertyRequest)
    # assertion triggered if 0 properties enumerated
    except AssertionError:
        print("0 properties found")
    # print("\nIteration {}: Found {} new properties".format(i, len(propertyFeatureExtractor.properties)))
    
    allProperties = propertyFeatureExtractor.properties
    print("--------------------------- Found {} properties -----------------------------".format(len(allProperties)))
    for prop in allProperties:
        print("Property: {}".format(prop))

    propertyTasks = convertToPropertyTasks(tasks, propertyRequest=propertyRequest)
    scoreModel = TaskDiscriminationScore(timeout=0.1, tasks=propertyTasks)
    for task, propertyTask in zip(tasks, propertyTasks):
        tasksForPropertyScoring = [t for t in propertyTasks if t != propertyTask]
        tasksPropertyScores = [(scoreModel.scoreProperty(prop, propertyTask, tasksForPropertyScoring), prop) for prop in allProperties]
        print("---------------------------------------------------------------------------------\n{}\n".format(task.describe()))
        sortedTaskPropertyScores = sorted(tasksPropertyScores, reverse=True, key=lambda x: x[0][1])
        for evalRes, prop in sortedTaskPropertyScores[:10]:
            print("Score: {:.2f} - Value: {} - {}".format(evalRes[1], evalRes[0], prop))

    if save:
        filename = "sampled_properties_{}_sampling_timeout={}s_seed={}.pkl".format(grammarName, int(featureExtractorArgs["propSamplingTimeout"]), seed)
        savePath = DATA_DIR + SAMPLED_PROPERTIES_DIR + filename
        dill.dump(allProperties, open(path, "wb"))
        print("Saving sampled properties at: {}".format(path))

    return properties
