import dill

from dreamcoder.type import arrow, tlist, tint, t0, UnificationFailure
from dreamcoder.utilities import eprint

def resume_from_path(resume):
    try:
        resume = int(resume)
        path = checkpointPath(resume)
    except ValueError:
        path = resume
    with open(path, "rb") as handle:
        result = dill.load(handle)
    resume = len(result.grammars) - 1
    eprint("Loaded checkpoint from", path)
    grammar = result.grammars[-1] if result.grammars else grammar
    return result, grammar, result.grammars[0]

def getLearnedProductions(result):
    currentProductions = set()
    learnedProductions = {}
    for i,grammar in enumerate(result.grammars):
        learnedProductions[i] = {}
        productions = set([str(p[2]) for p in grammar.productions])
        if len(list(currentProductions)) == 0:
            currentProductions = productions
        else:
            # print('----------------------------------------------------{}-----------------------------------------------------'.format(i))        
            newProductions = productions.difference(currentProductions)
            for j,production in enumerate(newProductions):
                learnedProductions[i][j] = production
                # print(j, production)
        currentProductions = currentProductions.union(productions)
    return learnedProductions


def getTrainFrontier(resumePath, n):
    result, resumeGrammar, firstGrammar = resume_from_path(resumePath)
    firstFrontier = [frontiers[0] for (key,frontiers) in result.frontiersOverTime.items() if len(frontiers[0].entries) > 0]
    allFrontiers = [frontier for (task,frontier) in result.allFrontiers.items() if len(frontier.entries) > 0]
    # expandedFrontier = expandFrontier(firstFrontier, n)
    print(result.learningCurve)

    testingTasks = result.getTestingTasks()
    trainTasks = result.taskSolutions.keys()

    learnedProductions = getLearnedProductions(result)
    return firstFrontier, allFrontiers, result.frontiersOverTime, resumeGrammar, firstGrammar, result.recognitionModel, learnedProductions, testingTasks, trainTasks


def scoreProgram(p, recognizer=None, grammar=None, task=None):

    if recognizer is not None:
        grammar = recognizer.grammarOfTask(task).untorch()

    ll = grammar.logLikelihood(task.request, p)
    return ll

def evaluateGrammars(frontiersOverTime, tasks, grammar1=None, grammar2=None, recognizer1=None, recognizer2=None):

    lastFrontier = {task:allFrontiers[-1] for task,allFrontiers in frontiersOverTime.items() if (len(allFrontiers[-1]) > 0) and task in tasks}

    print("{} Tasks have solutions for last frontier".format(len(lastFrontier.keys())))

    averageLLBefore, averageLLAfter = 0,0
    for task, frontier in lastFrontier.items():

        llBefore = scoreProgram(frontier.entries[0].program, task=task, grammar=grammar1, recognizer=recognizer1)
        llAfter = scoreProgram(frontier.entries[0].program, task=task, grammar=grammar2, recognizer=recognizer2)
        if llAfter == 0:
            pass
            # print("{}: {}".format(frontier.task, llBefore))
        else:
            pass
            # print("{}: {} -> {}".format(frontier.task.name, llBefore, llAfter))
        # print("Program: {}\n".format(frontier.entries[0].program))
        averageLLBefore += llBefore
        averageLLAfter += llAfter

    if llAfter == 0:
        print('Average LL: {}'.format(averageLLBefore / len(lastFrontier)))
    else:
        print('Average LL: {} -> {}'.format(averageLLBefore / len(lastFrontier), (averageLLAfter / len(lastFrontier))))

    return

# def trainRecognitionModel(featureExtractor, expandedFrontiers, timeout):

# timeout = 1200
# path = "recognitionModels/{}_trainTasks={}_timeout={}".format(datetime.datetime.now(), len(expandedFrontiers), timeout)
# trainedRecognizer = sleep_recognition(None, baseGrammar, [], [], [], expandedFrontiers, featureExtractor=featureExtractor, activation='tanh', CPUs=1, timeout=timeout, helmholtzFrontiers=[], helmholtzRatio=0, solver='ocaml', enumerationTimeout=0, skipEnumeration=True)
# with open(path,'wb') as handle:
#     dill.dump(trainedRecognizer, handle)
#     print('Stored recognizer at: {}'.format(path))



# trainedRecognizerPath = 'recognitionModels/2020-04-26 15:05:28.972185_trainTasks=2343_timeout=1200'
# with open(trainedRecognizerPath, 'rb') as handle:
#     trainedRecognizer = dill.load(handle)
# print('{} Train Tasks'.format(len(nnTrainTasks)))
# scoreTasks(trainedRecognizer, nnTrainTasks, taskToProgram, True)
# print('\n {} Test Tasks'.format(len(nnTestTasks)))
# scoreTasks(trainedRecognizer, nnTestTasks, taskToProgram, True)

def viewResults():
    # resumePath = 'kevinExperimentOutputs/list/'
    # resumeDirectory = '2019-03-21T15:55:48.767818/'
    # # resumeDirectory = '2021-03-23T18:37:24.574360/'
    # pickledFile = 'list_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=720_HR=0.5_it=19_MF=5_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=10_TRR=randomShuffle_K=2_topkNotMAP=False.pickle'
    # # pickledFile = 'list_arity=3_BO=False_CO=False_ES=1_ET=10_HR=0.5_it=20_MF=10_noConsolidation=True_RT=180_RR=False_RW=False_solver=ocaml_STM=True_TRR=default_K=2_topkNotMAP=False_DSL=False.pickle'
    # result, resumeGrammar, _ = resume_from_path(resumePath + resumeDirectory + pickledFile)

    baselinePickleFile = "kevinExperimentOutputs/list/2021-03-24T15:17:34.141045/list_arity=3_BO=False_CO=False_ES=1_ET=10_HR=0.5_it=20_MF=10_noConsolidation=True_RS=5000_RT=3600_RR=False_RW=False_solver=ocaml_STM=True_TRR=default_K=2_topkNotMAP=False_DSL=False.pickle"
    baselineResult, resumeGrammar, _ = resume_from_path(baselinePickleFile)

    baselinePlusPropSigPickleFile = "kevinExperimentOutputs/list/2021-04-02T19:22:26.195148/list_arity=3_BO=False_CO=False_ES=1_ET=10_HR=0.5_it=20_MF=10_noConsolidation=True_RS=5000_RT=3600_RR=False_RW=False_solver=ocaml_STM=True_TRR=default_K=2_topkNotMAP=False_DSL=False.pickle"
    baselinePlusPropSigResult, _, _ = resume_from_path(baselinePlusPropSigPickleFile)

    propSigOnlyPickleFile = "kevinExperimentOutputs/list/2021-04-02T18:49:05.106095/list_arity=3_BO=False_CO=False_ES=1_ET=10_HR=0.5_it=20_MF=10_noConsolidation=True_RS=5000_RT=3600_RR=False_RW=False_solver=ocaml_STM=True_TRR=default_K=2_topkNotMAP=False_DSL=False.pickle"
    propSigOnlyResult, resumeGrammar, _ = resume_from_path(propSigOnlyPickleFile)

    print(baselineResult.recognitionModel)
    print(propSigOnlyResult.recognitionModel)

    # # train tasks likelihood
    print("Train Tasks")
    evaluateGrammars(propSigOnlyResult.frontiersOverTime, propSigOnlyResult.taskSolutions.keys(), grammar1=resumeGrammar, grammar2=resumeGrammar, recognizer1=baselineResult.recognitionModel, recognizer2=baselinePlusPropSigResult.recognitionModel)

    # # test tasks likelihood
    print("Test Tasks")
    evaluateGrammars(propSigOnlyResult.frontiersOverTime, propSigOnlyResult.getTestingTasks(), grammar1=resumeGrammar, grammar2=resumeGrammar,  recognizer1=baselineResult.recognitionModel, recognizer2=baselinePlusPropSigResult.recognitionModel)

    # How does contextual model do?
    # evaluateGrammars(firstFrontier, manuallySolvedTasks, grammar1=topDownGrammar, recognizer2=resumeRecognizer)
