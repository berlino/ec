import dill
import numpy as np
import matplotlib.pyplot as plt

from dreamcoder.domains.list.compareProperties import compare

DATA_DIR = "data/prop_sig/"
ENUMERATION_RESULTS_DIR = "enumerationResults/"
SAMPLED_PROPERTIES_DIR = "sampled_properties/"

def plotFrontiers(fileNames, modelNames, save=True):
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.jet(np.linspace(0,1, len(fileNames))))

    for modelIdx,fileName in enumerate(fileNames):
        frontiers, times = dill.load(open(ENUMERATION_RESULTS_DIR + fileName, "rb"))

        satisfiesHoldout = lambda f: f.task.check(f.topK(1).entries[0].program, timeout=1.0, leaveHoldout=False)
        logPosteriors = sorted([-f.bestPosterior.logPosterior for f in frontiers if (len(f.entries) > 0 and satisfiesHoldout(f))])
        print(fileName, len(logPosteriors))
        plt.plot(logPosteriors, [i / len(frontiers) for i in range(len(logPosteriors))], label=modelNames[modelIdx], alpha=0.6)

    plt.ylim(bottom=0, top=1)
    plt.legend()
    plt.show()
    if save:
        plt.savefig("enumerationResults/enumerationTimes.png")
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