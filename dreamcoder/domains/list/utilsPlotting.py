import dill
import numpy as np
import matplotlib.pyplot as plt

from dreamcoder.domains.list.compareProperties import compare

DATA_DIR = "data/prop_sig/"
ENUMERATION_RESULTS_DIR = "enumerationResults/"
SAMPLED_PROPERTIES_DIR = "sampled_properties/"

def plotFrontiers(modelNames, fileNames=None, enumerationResults=None, save=True, plotName="enumerationTimes"):

    if enumerationResults is None:
        if fileNames is None:
            assert Exception("You must either provide the filenames of the pickled enumeration results or the results themselves")
        else:
            enumerationResults = [(frontiers, times) for fileName in dill.load(open(ENUMERATION_RESULTS_DIR + fileName, "rb"))]    

    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.jet(np.linspace(0,1, len(modelNames))))

    for enumerationResult, modelName in zip(enumerationResults, modelNames):
        
        frontiers, times = enumerationResult
        satisfiesHoldout = lambda f: f.task.check(f.topK(1).entries[0].program, timeout=1.0, leaveHoldout=False)

        nonEmptyFrontiers = [f for f in frontiers if len(f.entries) > 0]
        logPosteriors = sorted([-f.bestPosterior.logPosterior for f in nonEmptyFrontiers if satisfiesHoldout(f)])
        print("{}: {} / {}".format(modelName, len(logPosteriors), len(nonEmptyFrontiers)))
        plt.plot(logPosteriors, [i / len(frontiers) for i in range(len(logPosteriors))], label=modelName, alpha=0.6)

    plt.ylim(bottom=0, top=1)
    plt.legend()
    plt.show()
    if save:
        plt.savefig("enumerationResults/{}.png".format(plotName))
    return

def plotProxyResults(modelToLogPosteriors, plotName="enumerationProxy", save=True):

    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.jet(np.linspace(0,1, len(modelToLogPosteriors.keys()))))

    for modelName, task2logPosteriors in modelToLogPosteriors.items():
        logPosteriors = sorted([-lp for _,lp in task2logPosteriors.items()])
        plt.plot(logPosteriors, [i / len(logPosteriors) for i in range(len(logPosteriors))], label=modelName, alpha=0.6)

    plt.ylim(bottom=0, top=1)
    plt.legend()
    plt.show()
    if save:
        plt.savefig("enumerationResults/{}.png".format(plotName))
    return


def plotNumSampledPropertiesVersusMdl(propertiesFilename):
    enumeratedProperties = dill.load(open(DATA_DIR + SAMPLED_PROPERTIES_DIR + propertiesFilename, "rb"))
    assert len(enumeratedProperties.keys()) == 1
    allProperties = list(enumeratedProperties.values())[0]
    logPriors = sorted([-p.logPrior for p in allProperties])
    plt.plot(np.exp(logPriors), np.arange(0,len(logPriors)))
    plt.ylabel("Number of Unique Tasks Signature properties found")
    plt.xlabel("Enumeration Time")
    plt.show()

def plotNumSampledPropertiesVersusTimeout(timeouts, propertiesFilenames, handwrittenProperties, tasks, sampledFrontiers, valuesToInt):
    numHandwrittenDiscoveredArr = []
    numTotalArr = []
    for t,filename in zip(timeouts, propertiesFilenames):
        
        enumeratedProperties = dill.load(open(DATA_DIR + SAMPLED_PROPERTIES_DIR + filename, "rb"))
        assert len(enumeratedProperties.keys()) == 1
        allProperties = enumeratedProperties[tasks[0]]
        numTotalArr.append(len(allProperties))

        equivalentSampledProperties = compare(handwrittenProperties, allProperties, tasks, sampledFrontiers, valuesToInt)
        numHandwrittenDiscoveredArr.append(len(equivalentSampledProperties))

    plt.plot(timeouts, np.array(numHandwrittenDiscoveredArr) / len(handwrittenProperties))
    plt.ylabel("Percent of Handwritten properties discovered")
    plt.ylim([0, 1])
    plt.xlabel("Property Enumeration Time (s)")
    plt.show()
    return

# handwrittenProperties = getHandwrittenPropertiesFromTemplates(tasks)
# valuesToInt = {"allFalse":0, "allTrue":1, "mixed":2}
# timeouts = [1,5,10,30,60,120,180,300,3600]
# propertiesFilenames = ["sampled_properties_weights=fitted_sampling_timeout={}s_return_types=[bool]_seed=1.pkl".format(t) for t in timeouts]
# plotNumSampledPropertiesVersusTimeout(timeouts, propertiesFilenames, handwrittenProperties, tasks, sampledFrontiers, valuesToInt)
# plotNumSampledPropertiesVersusMdl(propertiesFilenames[-1])
