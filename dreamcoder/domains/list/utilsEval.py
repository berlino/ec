import datetime
import dill

from dreamcoder.domains.list.utilsPlotting import plotFrontiers
from dreamcoder.enumeration import multicoreEnumeration
from dreamcoder.grammar import Grammar
from dreamcoder.utilities import vprint

def getRecognizerTaskGrammars(trainedRecognizer, tasks):
    grammars = {task: trainedRecognizer.grammarOfTask(task)
                for task in tasks}
    #untorch seperately to make sure you filter out None grammars
    grammars = {task: grammar.untorch() for task, grammar in grammars.items() if grammar is not None}
    return grammars


def enumerateFromGrammar(grammars, tasks, modelName, enumerationTimeout, solver, CPUs, maximumFrontier, leaveHoldout=False, save=False):
    bottomUpFrontiers, allRecognitionTimes, _, _ = multicoreEnumeration(grammars, tasks, _=None,
                             enumerationTimeout=enumerationTimeout,
                             solver=solver,
                             CPUs=CPUs,
                             maximumFrontier=maximumFrontier,
                             verbose=True,
                             evaluationTimeout=1.0,
                             testing=False,
                             likelihoodModel=None,
                             leaveHoldout=leaveHoldout)
    if save:
        savePath = "enumerationResults/{}_{}_t={}.pkl".format(modelName, datetime.datetime.now(), enumerationTimeout)
        with open(savePath, "wb") as handle:
            dill.dump((bottomUpFrontiers, allRecognitionTimes), handle)
        print("Saved enumeration results at: {}".format(savePath))
    return bottomUpFrontiers, allRecognitionTimes

def enumerateFromGrammars(args, tasks, allGrammars, modelNames, save, plotName):

    enumerationTimeout, solver, maximumFrontier, CPUs = args.pop("enumerationTimeout"), args.pop("solver"), args.pop("maximumFrontier"), args.pop("CPUs")
    
    enumerationResults = []
    for g, modelName in zip(allGrammars, modelNames):
         print("grammar for first task: {}".format(g if isinstance(g, Grammar) else list(g.values())[0]))
         bottomUpFrontiers, allRecognitionTimes = enumerateFromGrammar(g, tasks, modelName, enumerationTimeout, solver, CPUs, maximumFrontier, leaveHoldout=True, save=save)
         enumerationResults.append((bottomUpFrontiers, allRecognitionTimes))
         nonEmptyFrontiers = [f for f in bottomUpFrontiers if not f.empty]
         numTasksSolved = len([f.task for f in nonEmptyFrontiers if f.task.check(f.topK(1).entries[0].program, timeout=1.0, leaveHoldout=False)])
         print("Enumerating from {} grammars for {} seconds: {} / {} actually true for holdout example".format(modelName, enumerationTimeout, numTasksSolved, len(nonEmptyFrontiers)))
    
    plotFrontiers(modelNames, enumerationResults=enumerationResults, save=True, plotName=plotName)
    return

def enumerationProxy(task2FittedGrammars, tasks, modelNames, verbose=False):
    """
    Prints the log posterior of tasks under different grammars

    Args:
        task2FittedGrammars (list): List of either task2grammar dictionaries or Grammar if grammar is the same for all tasks
        tasks (list(Task)): List of tasks for which to calculate log posterior
        modelNames (list(str)): List of model names corresponding to task2FittedGrammars

    Returns:
        None
    """

    modelToLogPosteriors = {modelName:[] for modelName in modelNames}
    tasksWithGtPrograms = [t for t in tasks if t.program is not None]
    for task in tasksWithGtPrograms:
        vprint("\n-------------------------------------------------------------------------------", verbose)
        vprint(task.describe(), verbose)
        vprint("Ground Truth Program: {}".format(task.program), verbose)
        vprint("---------------------------------------------------------------------------------", verbose)
        for task2grammar, modelName in zip(task2FittedGrammars, modelNames):
            taskGrammar = task2grammar if isinstance(task2grammar, Grammar) else task2grammar[task]
            logPosterior = taskGrammar.logLikelihood(task.request, task.program)
            modelToLogPosteriors[modelName].append(logPosterior)
            vprint("{}: {}".format(modelName, logPosterior), verbose)

    for modelName, logPosteriors in modelToLogPosteriors.items():
        vprint("Mean {} Log posterior: {}".format(modelName, sum(logPosteriors) / len(logPosteriors)), verbose)

    return {modelName: {t: logPosterior for t,logPosterior in zip(tasksWithGtPrograms, logPosteriors)} for modelName,logPosteriors in modelToLogPosteriors.items()}

# def enumerationProxy(task2FittedGrammar, train, grammar, nSim, task2groundTruthPrograms=None, neuralBaselinePath=NEURAL_RECOGNITION_MODEL_PATH, verbose=False):
#     """
#     Given a frontier of tasks prints out the logposterior of tasks in train using:
#         - the RNN-encoded neural recogntion model
#         - the unigram grammar fitted on nSim most similar tasks
#     """

#     if task2groundTruthPrograms is not None:
#         raise NotImplementedError

#     recognitionModel = dill.load(open(NEURAL_RECOGNITION_MODEL_PATH, "rb"))

#     uniformGrammarPriors, logVariableGrammarPriors, fittedLogPosteriors, neuralRecognitionLogPosteriors = 0.0, 0.0, 0.0, 0.0

#     numTasks = 0
#     fittedTasks = None
#     if isinstance(task2FittedGrammar, list):
#         fittedTasks = list(task2FittedGrammar[0].keys())
#     else:
#         fittedTasks = list(task2FittedGrammar.keys())
#     for task in train:
#         # if we have a fitted grammar for this task and a ground truth program we can score the grammar on
#         if task in fittedTasks and task.program is not None:
#             numTasks += 1
#             vprint("\n-------------------------------------------------------------------------------", verbose)
#             vprint(task.describe(), verbose)
#             vprint("Ground Truth Program: {}".format(task.program), verbose)
#             vprint("---------------------------------------------------------------------------------", verbose)
#             uniformGrammarPrior = grammar.logLikelihood(task.request, task.program)
#             vprint("Uniform Grammar Prior: {}".format(uniformGrammarPrior), verbose)
#             logVariableGrammar = Grammar(2.0, [(0.0, p.infer(), p) for p in grammar.primitives], continuationType=None)
#             logVariableGrammarPrior = logVariableGrammar.logLikelihood(task.request, task.program)
#             vprint("Log Variable Program Prior: {}".format(logVariableGrammarPrior), verbose)
#             vprint("---------------------------------------------------------------------------------", verbose)

#             if isinstance(task2FittedGrammar, list):
#                 toPrint = "PropSim Grammar LP ({} frontiers):".format(nSim)
#                 for task2Grammar in task2FittedGrammar:
#                     if task in task2Grammar:     
#                         fittedLogPosterior = task2Grammar[task].logLikelihood(task.request, task.program)
#                         toPrint += " {} ->".format(fittedLogPosterior)
#                     else:
#                         toPrint += "solved"
#                 vprint(toPrint, verbose)
#             else:
#                 if task in task2FittedGrammar:
#                     fittedLogPosterior = task2FittedGrammar[task].logLikelihood(task.request, task.program)
#                     vprint("ProSim Grammar LP ({} frontier): {}".format(nSim, fittedLogPosterior), verbose) 
            
#             neuralGrammar = recognitionModel.grammarOfTask(task)
#             neuralRecognitionLogPosterior = neuralGrammar.logLikelihood(task.request, task.program).item()
#             vprint("Neural Recognition LogPosterior: {}\n".format(neuralRecognitionLogPosterior), verbose)
            
#             uniformGrammarPriors += uniformGrammarPrior
#             logVariableGrammarPriors += logVariableGrammarPrior
#             neuralRecognitionLogPosteriors += neuralRecognitionLogPosterior
#             fittedLogPosteriors += fittedLogPosterior

#     if numTasks == 0:
#         print("No solved frontiers from which to report metrics")
#         return
#     else:
#         print("Metrics for {} tasks with solutions".format(numTasks))

#     print("Mean Uniform Grammar Prior: {}".format(uniformGrammarPriors / numTasks))
#     print("Mean Log Variable Grammar Prior: {}".format(logVariableGrammarPriors / numTasks))
#     print("Neural Recognition Log Posterior: {}".format(neuralRecognitionLogPosteriors / numTasks))
#     print("Mean Fitted Log Posterior ({} frontiers): {}".format(nSim, fittedLogPosteriors / numTasks))

#     return task2FittedGrammar
