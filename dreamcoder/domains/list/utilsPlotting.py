import dill
import numpy as np
import matplotlib.pyplot as plt
import pickle

from dreamcoder.domains.list.compareProperties import compare
from dreamcoder.domains.list.utilsEval import cumulativeNumberOfTasksSolved, loadEnumerationResults

DATA_DIR = "data/prop_sig/"
ENUMERATION_RESULTS_DIR = "enumerationResults/"
SAMPLED_PROPERTIES_DIR = "sampled_properties/"
ENUMERATION_TIME = 600

def plotFrontiers(modelNames, filenames=None, times=False, enumerationResults=None, save=True, plotName="enumerationTimes"):

    if enumerationResults is None:
        if filenames is None:
            assert Exception("You must either provide the filenames of the pickled enumeration results or the results themselves")
        else:
            enumerationResults = [loadEnumerationResults(filename) for filename in filenames]

    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.jet(np.linspace(0,1, len(modelNames))))

    for enumerationResult, modelName in zip(enumerationResults, modelNames):
        
        frontiers, times = enumerationResult
        numTasks = len(frontiers)

        satisfiesHoldout = lambda f: f.task.check(f.topK(1).entries[0].program, timeout=1.0, leaveHoldout=False)
        nonEmptyFrontiers = [f for f in frontiers if len(f.entries) > 0]
        logPosteriors = sorted([-f.bestPosterior.logPosterior for f in nonEmptyFrontiers if satisfiesHoldout(f)])
        print("{}: {} / {}".format(modelName, len(logPosteriors), len(nonEmptyFrontiers)))

        satisfiesHoldout = lambda f: len(f.entries) > 0 and f.task.check(f.topK(1).entries[0].program, timeout=1.0, leaveHoldout=False)
        solvedTasks = set(f.task for f in frontiers if satisfiesHoldout(f))
        if times:
            toPlot = sorted([time for task,time in times.items() if task in solvedTasks])
            percentTasksSolved = [i / numTasks for i in range(len(toPlot))]
            # adding point to the end so that plot lines extend all the way to the far right side of the plot
            plt.plot(toPlot + [ENUMERATION_TIME], percentTasksSolved + [percentTasksSolved[-1]], label=modelName, alpha=0.6)
        else:
            toPlot = sorted([-f.bestPosterior.logPosterior for f in frontiers if f.task in solvedTasks])
            plt.plot(toPlot, [i / numTasks for i in range(len(toPlot))], label=modelName, alpha=0.6)

    plt.ylim(bottom=0, top=1)
    plt.xlabel("Enumeration time (s)")
    plt.ylabel("Percent tasks solved")
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

def barPlotPerTaskStatistics(modelNames, enumerationFilenames, grammarFilenames=None, plotTime=True, includeUnsolved=False):
    if includeUnsolved:
        if grammarFilenames is None:
            raise Exception("You must provided the fitted grammars if you want to include the unsolved tasks in the plot")
        else:
            grammars = [loadEnumerationResults(filename) for filename in grammarFilenames]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.jet(np.linspace(0,1, len(modelNames))))

    frontiers, times = loadEnumerationResults(enumerationFilenames[0])

    if plotTime:
        toPlot = [(task,time) if time is not None else (task, -10) for task,time in times.items()]
    else:
        toPlot = [(f.task, -f.topK(1).entries[0].logPosterior) if len(f.entries) > 0 is not None 
        else (f.task, (-grammars[0][f.task].logLikelihood(f.task.request, f.task.program) if includeUnsolved else 0)) for f in frontiers]
    
    sortedTimes = sorted(toPlot, key=lambda x: x[1])
    tasks, toPlot = list(zip(*sortedTimes))
    taskNames = [t.name for t in tasks]

    # Calculate optimal width
    width = (1.0 / float(len(enumerationFilenames))) / 2.0
    ax.bar([x_coord for x_coord in range(len(tasks))], toPlot, width, label=modelNames[0], align="edge")

    offset = width
    for i in range(1, len(modelNames)):
        modelName, enumerationFilename = modelNames[i], enumerationFilenames[i]
        frontiers, times = loadEnumerationResults(enumerationFilename)
        task2frontier = {f.task:f for f in frontiers}

        if plotTime:
            toPlot = [times[task] if times[task] is not None else -10 for task in tasks]
        else:
            toPlot = [-task2frontier[task].topK(1).entries[0].logPosterior if len(task2frontier[task].entries) > 0 is not None 
            else (-grammars[i][task].logLikelihood(task.request, task.program) if includeUnsolved else 0) for task in tasks]

        ax.bar([x_coord + offset for x_coord in range(len(tasks))], toPlot, width, label=modelName, align="edge")
        offset += width

    ax.axes.set_xticklabels(taskNames)
    ax.set_ylabel('Negative log probability')
    ax.set_xlabel('Concepts')
    plt.xticks(rotation=90, fontsize=6)
    plt.legend()
    plt.show()

    return

def main():
    modelNames, enumerationFilenames, grammarFilenames = zip(*[
        ############ Random primitive weights (seed=1) #############
        # "propsim-fitted": "propsimGrammarsHandwritten_2022-01-01_23:19:30.389021_t=600.pkl",
        # # "propsimGrammarsHandwrittenEqWeight": "propsimGrammarsHandwrittenEqWeight_2022-01-01_23:36:17.413812_t=600.pkl",
        # # "propsimGrammarsAutomatic_2022": "propsimGrammarsAutomatic_2022-01-02_00:10:34.733461_t=600.pkl",
        # # "propsimGrammarsAutomaticEqWeight_2022-01-01_23:53:34.640085_t=600.pkl",
        # "all-fitted": "helmholtzFitted_2022-01-02_00:20:35.402688_t=600.pkl",

        ############ Equal weight primitives #############
        ("PropsimFit", "propsimGrammarsHandwritten_2022-01-01_22:14:36.614585_t=600.pkl", "data/prop_sig/helmholtz_frontiers/josh_rich_0_10_enumerated/13742_with_josh_fleet_0_10-inputs__propSim_propToUse=handwritten_numHelmFrontiers_10000_nSim=50_weightedSim=False_onlyTrueProp=_False_taskSpecificInputs=True_compressSimilar=False_equalWprop=False_seed=2_grammars.pkl"), 
        # "propsimGrammarsHandwrittenEqWeight_2022-01-01_22:30:59.723239_t=600.pkl",
        # "propsimGrammarsAutomaticEqWeight_2022-01-01_22:47:51.005722_t=600.pkl",
        # "propsimGrammarsAutomatic_2022-01-01_23:04:27.669035_t=600.pkl",
        ("AllFit", "helmholtzFitted_2022-01-04_13:45:58.645759_t=600.pkl", "data/prop_sig/helmholtz_frontiers/josh_rich_0_10_enumerated/13742_with_josh_fleet_0_10-inputs__helmholtz_10000_grammars.pkl"),
        ("Uniform", "uniform_2022-01-04_14:07:30.300207_t=600.pkl", None)
    ])

    # grammarFilenames = [
    #     "propsimGrammarsHandwritten_2022-01-03_22:02:28.760806_t=5.pkl",
    #     "neuralGrammars_2022-01-03_22:02:40.586424_t=5.pkl",
    #     "helmholtzGrammar_2022-01-03_22:02:45.937921_t=5.pkl"
    # ]

    for i in [10,25,50,100,600]:
        print("-------------------------------------------------------")
        for modelName,enumerationFilename in zip(modelNames, enumerationFilenames):
            print("{}: {}".format(modelName, cumulativeNumberOfTasksSolved(enumerationFilename, i)))

    plotFrontiers(modelNames, filenames=enumerationFilenames, times=True, enumerationResults=None, save=True, plotName="enumerationTimes")

    # barPlotPerTaskStatistics(modelNames, enumerationFilenames, grammarFilenames, plotTime=True, includeUnsolved=False)
    return 


# handwrittenProperties = getHandwrittenPropertiesFromTemplates(tasks)
# valuesToInt = {"allFalse":0, "allTrue":1, "mixed":2}
# timeouts = [1,5,10,30,60,120,180,300,3600]
# propertiesFilenames = ["sampled_properties_weights=fitted_sampling_timeout={}s_return_types=[bool]_seed=1.pkl".format(t) for t in timeouts]
# plotNumSampledPropertiesVersusTimeout(timeouts, propertiesFilenames, handwrittenProperties, tasks, sampledFrontiers, valuesToInt)
# plotNumSampledPropertiesVersusMdl(propertiesFilenames[-1])
