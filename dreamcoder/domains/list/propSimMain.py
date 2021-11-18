from dreamcoder.domains.list.propSim import *
from dreamcoder.domains.list.runUtils import *
from dreamcoder.domains.list.utilsBaselines import *
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
                sampledFrontiers[t] = enumerateHelmholtzOcaml(tasks, task2FittedGrammar[t], args["enumerationTimeout"], args["CPUs"], featureExtractor, save=False, libraryName=args["libraryName"], dataset=dataset)
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
             directory = DATA_DIR + "grammars/{}_primitives/enumerated_{}:{}".format(args["libraryName"], args["hmfSeed"], args["helmholtzFrontiersFilename"].split(":")[0])
             directory += ":{}/".format(args["numHelmFrontiers"]) if args["numHelmFrontiers"] is not None else "/"
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

    # Enumeration
    # helmholtzGrammar = baseGrammar.insideOutside(initSampledFrontiers, 1, iterations=1, frontierWeights=None, weightByPrior=False)
    # uniformGrammar = baseGrammar
    # neuralGrammars = getGrammarsFromNeuralRecognizer(LearnedFeatureExtractor, tasks, baseGrammar, {"hidden": args["hidden"]}, initSampledFrontiers, args["save"], directory, args)
    # allGrammars = [uniformGrammar, task2FittedGrammar]
    # modelNames = ["uniformGrammar", "PropSim"]
    # enumerate_from_grammars(args)

    tasks = get_tasks(args["dataset"])
    tasks = tasks[0:1] if args["singleTask"] else tasks
    prims = get_primitives(args["libraryName"])
    baseGrammar = Grammar.uniform([p for p in prims])

    if args["libraryName"] == "josh_rich":
        # now that we've loaded the primitives we can parse the ground truth program string
        for t in tasks:
            t.parse_program(prims)

    # get frontiers to fit on
    # frontiers = enumerateHelmholtzOcaml(tasks, baseGrammar, enumerationTimeout=180, CPUs=40, featureExtractor=featureExtractor, save=True, libraryName=args["libraryName"], dataset=args["dataset"], saveDirectory=None)
    sampledFrontiers = loadEnumeratedTasks(dslName=args["libraryName"], filename=args["helmholtzFrontiersFilename"], 
       primitives=prims, hmfSeed=args["hmfSeed"])[:10000]
   
    featureExtractor, properties = get_extractor(tasks, baseGrammar, args) 
    propsimGrammars = iterative_propsim(args, tasks, baseGrammar, properties, sampledFrontiers)
    # editDistGrammars = getGrammarsFromEditDistSim(tasks, baseGrammar, sampledFrontiers, args["nSim"])
    enumerationProxy(propsimGrammars, tasks, baseGrammar, args["nSim"], verbose=True)
    return
