from dreamcoder.domains.list.propSim import *
from dreamcoder.domains.list.runUtils import *
from dreamcoder.domains.list.utilsBaselines import *
from dreamcoder.domains.list.utilsPlotting import plotProxyResults
from dreamcoder.domains.list.utilsProperties import *

VALUES_TO_INT = {"allFalse":0, "allTrue":1, "mixed":2}

def iterative_propsim(args, tasks, baseGrammar, properties, initSampledFrontiers):

    #################################
    # Load sampled tasks
    #################################

    taskFittedGrammars = []
    for propSimIteration in range(args["propNumIters"]):
        print("\nLoading helmholtz tasks for iteration {}".format(propSimIteration))

        if propSimIteration == 0:
            tasksToSolve = tasks
            sampledFrontiers = {t: initSampledFrontiers for t in tasksToSolve}
            
            task2FittedGrammar = {t:baseGrammar for t in tasksToSolve}
            print("Attempting to solve tasks: {}".format("\n".join([str(t) for t in tasksToSolve])))

        else:
            sampledFrontiers = {}
            for t in tasksToSolve:
                sampledFrontiers[t] = enumerateHelmholtzOcaml(tasks, task2FittedGrammar[t], args["enumerationTimeout"], args["CPUs"], featureExtractor, save=False)
                print("Enumerated {} helmholtz tasks for task {}".format(len(sampledFrontiers[t]), t))
        # use subset (numHelmFrontiers) of helmholtz tasks
        for t in tasksToSolve:
            if args["numHelmFrontiers"] is not None and args["numHelmFrontiers"] < len(sampledFrontiers[t]):
                sampledFrontiers[t] = sorted(sampledFrontiers[t], key=lambda f: f.topK(1).entries[0].logPosterior, reverse=True)
                sampledFrontiers[t] = sampledFrontiers[t][:min(len(sampledFrontiers[t]), args["numHelmFrontiers"])]

            print("Finished loading {} helmholtz tasks for task {}".format(len(sampledFrontiers[t]), str(t)))

        #################################
        # Get Grammars
        #################################

        try:
             propSimFilename = "propSim_propToUse={}_nSim={}_weightedSim={}_taskSpecificInputs={}_seed={}.pkl".format(
                args["propToUse"], args["nSim"], args["weightedSim"], args["taskSpecificInputs"], args["seed"])
             # directory = DATA_DIR + "grammars/{}_primitives/enumerated_{}:{}".format(args["libraryName"], args["hmfSeed"], args["helmholtzFrontiers"].split(":")[0])
             # directory += ":{}/".format(args["numHelmFrontiers"]) if args["numHelmFrontiers"] is not None else "/"
             directory = DATA_DIR
             path = directory + propSimFilename
             propSimGrammars = dill.load(open(path, "rb"))
        except FileNotFoundError:
             print("Couldn't find pickled fitted grammars, regenerating")

        task2FittedGrammar, tasksSolved, _ = getPropSimGrammars(
           baseGrammar,
           tasksToSolve,
           tasks, 
           sampledFrontiers, 
           properties, 
           args["onlyUseTrueProperties"], 
           args["nSim"], 
           args["propPseudocounts"], 
           args["weightedSim"], 
           compressSimilar=False, 
           weightByPrior=False, 
           recomputeTasksWithTaskSpecificInputs=args["taskSpecificInputs"],
           computePriorFromTasks=args["computePriorFromTasks"], 
           filterSimilarProperties=args["filterSimilarProperties"], 
           maxFractionSame=args["maxFractionSame"], 
           valuesToInt=VALUES_TO_INT,
           propSimIteration=propSimIteration,
           verbose=args["verbose"])

        taskFittedGrammars.append(task2FittedGrammar)

        print("\nSolved {} tasks at iteration {}".format(len(tasksSolved), propSimIteration))
        fileName = "enumerationResults/propSim_2021-06-28 19:33:34.730379_t=1800.pkl"
        frontiers, times = dill.load(open(fileName, "rb"))

        tasksToSolve = [t for t in tasksToSolve if t not in tasksSolved]
        print("{} still unsolved\n".format(len(tasksToSolve)))
        if len(tasksToSolve) == 0:
            break

    return taskFittedGrammars[0]

def enumerate_from_grammars(args, allGrammars, modelNames):

    enumerationTimeout, solver, maximumFrontier, CPUs = args.pop("enumerationTimeout"), args.pop("solver"), args.pop("maximumFrontier"), args.pop("CPUs")

    for g, modelName in zip(allGrammars, modelNames):
         print("grammar for first task: {}".format(g if isinstance(g, Grammar) else list(g.values())[0]))
         bottomUpFrontiers, allRecognitionTimes = enumerateFromGrammars(g, tasks, modelName, enumerationTimeout, solver, CPUs, maximumFrontier, leaveHoldout=True, save=save)
         nonEmptyFrontiers = [f for f in bottomUpFrontiers if not f.empty]
         numTasksSolved = len([f.task for f in nonEmptyFrontiers if f.task.check(f.topK(1).entries[0].program, timeout=1.0, leaveHoldout=False)])
         print("Enumerating from {} grammars for {} seconds: {} / {} actually true for holdout example".format(modelName, enumerationTimeout, numTasksSolved, len(nonEmptyFrontiers)))

    return

def main(args):
       
    print("cuda: {}".format(torch.cuda.is_available())) 
    # Enumeration
    # helmholtzGrammar = baseGrammar.insideOutside(initSampledFrontiers, 1, iterations=1, frontierWeights=None, weightByPrior=False)
    # uniformGrammar = baseGrammar
    # neuralGrammars = getGrammarsFromNeuralRecognizer(LearnedFeatureExtractor, tasks, baseGrammar, {"hidden": args["hidden"]}, initSampledFrontiers, args["save"], directory, args)
    # allGrammars = [uniformGrammar, task2FittedGrammar]
    # modelNames = ["uniformGrammar", "PropSim"]
    # enumerate_from_grammars(args)

    # Load tasks, DSL, features extractor and properties
    tasks = get_tasks(args["dataset"])
    tasks = tasks[0:1] if args["singleTask"] else tasks
    prims = get_primitives(args["libraryName"])
    baseGrammar = Grammar.uniform([p for p in prims])
    featureExtractor, properties = get_extractor(tasks, baseGrammar, args) 

    if "josh_rich" in args["libraryName"]:
        # now that we've loaded the primitives we can parse the ground truth program string
        for t in tasks:
            # parses program string and also executes to check that I/O matches parsed program
            t.parse_program(prims)

    # get helmholtz frontiers either by loading saved file, or by enumerating new ones
    if args["helmholtzFrontiers"] is not None: 
        datasetName = args["helmholtzFrontiers"][:args["helmholtzFrontiers"].index(".pkl")]
        dslDirectory, pklName = args["helmholtzFrontiers"].split("/")
        helmholtzFrontiers = loadEnumeratedTasks(filename=args["helmholtzFrontiers"], primitives=prims)
    else:
        datasetName = args["dataset"]
        helmholtzFrontiers = enumerateHelmholtzOcaml(tasks, baseGrammar, enumerationTimeout=1800, CPUs=40, featureExtractor=featureExtractor, save=True, libraryName=args["libraryName"], datasetName=datasetName)    

    saveDirectory = "{}helmholtz_frontiers/{}/".format(DATA_DIR, dslDirectory)
    testingTasks = get_tasks("josh_fleet_0_99")
    neuralGrammars = getGrammarsFromNeuralRecognizer(LearnedFeatureExtractor, tasks, testingTasks, baseGrammar, {"hidden": args["hidden"]}, helmholtzFrontiers, args["save"], saveDirectory, datasetName, args)
 
    featureExtractor, properties = get_extractor(tasks, baseGrammar, args) 
    propsimGrammars = iterative_propsim(args, tasks, baseGrammar, properties, helmholtzFrontiers)
    # editDistGrammars = getGrammarsFromEditDistSim(tasks, baseGrammar, sampledFrontiers, args["nSim"])

    helmholtzGrammar = baseGrammar.insideOutside(helmholtzFrontiers, pseudoCounts=1)
    grammars = [neuralGrammars, propsimGrammars, helmholtzGrammar, baseGrammar]
    modelNames = ["neural", "propsim", "helmholtz", "uniform"]
    modelToLogPosteriors = enumerationProxy(grammars, tasks, modelNames, verbose=True)
    plotProxyResults(modelToLogPosteriors, save=False)
    # enumerate_from_grammars(args, [propSimGrammars, editDistGrammars], ["propSimGrammars", "editDistGrammars"])
    return
