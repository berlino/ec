import dill

from dreamcoder.domains.list.listPrimitives import basePrimitives, primitives, McCarthyPrimitives, bootstrapTarget_extra, no_length, josh_primitives
from dreamcoder.domains.list.taskProperties import handWrittenProperties, tinput, toutput
from dreamcoder.grammar import Context, Grammar
from dreamcoder.program import *
from dreamcoder.type import *
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
    if p is None:
        return -100

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

def evaluateRecognizers(grammar, ecResults, recognizerNames=None, iteration=-1):

    if recognizerNames is None:
        recognizerNames = [str(i) for i in range(len(ecResults))]

    tasks = list(ecResults[0].frontiersOverTime.keys())
    
    for task in tasks:

        print("----------------------------------------------------------------------------------------------------------------------------------")
        print("\n\nTask {}".format(task))
        for i in range(min(len(task.examples), 10)):
            print("{} -> {}".format(task.examples[i][0][0], task.examples[i][1]))
        print("----------------------------------------------------------------------------------------------------------------------------------")
        programs = []
        for i,ecResult in enumerate(ecResults):
            bestFrontier = ecResult.frontiersOverTime[task][iteration].topK(1)
            if len(bestFrontier) > 0:
                program = bestFrontier.entries[0].program
            else:
                program = None
            programs.append(program)
        
        if all([p is None for p in programs]):
            continue
        else:
            bestProgram = max([(program,scoreProgram(program, ecResult.recognitionModel, grammar=grammar, task=task)) for program in programs],
                key=lambda x: x[1])[0]

            bestPriorProgram, logPrior = max([(program,scoreProgram(program, None, grammar=grammar, task=task)) for program in programs],
                key=lambda x: x[1])

            print("Best Program LogPrior: {} -> {}".format(bestPriorProgram, logPrior))
            print("----------------------------------------------------------------------------------------------------------------------------------")
            for i,p in enumerate(programs):
                logPosterior = scoreProgram(bestProgram if p is None else p, ecResults[i].recognitionModel, grammar=grammar, task=task)
                logPosteriorStr = str(logPosterior) if p is not None else "bestProgramLogPosterior: {}".format(logPosterior)
                print("{}: {} ({})".format(recognizerNames[i], p, logPosteriorStr))

    return

def evaluateRecognizersForTask(grammar, ecResults, recognizerNames=None, taskToInvestigate=None, request=None, productionToInvestigate=None, printGrammar=False):

    print("\nGrammars\n--------------------------------------------------------------------------------------------------------------------------")

    if recognizerNames is None:
        recognizerNames = [str(i) for i in range(len(ecResults))]

    tasks = list(ecResults[0].frontiersOverTime.keys())
    task = [t for t in tasks if t.name == taskToInvestigate][0]

    grammars = []
    for i, ecResult in enumerate(ecResults):
        grammar = ecResult.recognitionModel.grammarOfTask(task).untorch()
        grammars.append(grammar)

        if request is not None:
            table = grammar.buildCandidates(request, Context.EMPTY, [], normalize=True, returnTable=True, returnProbabilities=True, mustBeLeaf=False)
            table = {key.name:value for key,value in table.items()}
            print("\nRequest: {}, Recognizer: {}".format(request, recognizerNames[i]))
            if productionToInvestigate is not None:
                print("p({}): {}".format(productionToInvestigate, table[productionToInvestigate][0]))
            else:
                print(table)

    if printGrammar:
        print("\n")
        for i,el in enumerate(grammar.productions):
            _,_,production = el

            toDisplay = "{}: {:.2f} (logPrior) ".format(production, float(grammar.productions[i][0]))
            for j,grammar in enumerate(grammars):
                toDisplay += "| {:.2f} ({}) ".format(float(grammar.productions[i][0]), recognizerNames[j])
            print(toDisplay)

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

    baselinePickleFile = "experimentOutputs/jrule/2021-04-16T19:26:16.859630/jrule_arity=3_BO=False_CO=False_dp=False_doshaping=False_ES=1_ET=5_epochs=9999_HR=1.0_it=1_MF=10_parallelTest=False_RT=3600_RR=False_RW=False_st=False_STM=True_TRR=default_K=2_topkNotMAP=False_tset=S12_DSL=False.pickle"
    baselineResult, resumeGrammar, _ = resume_from_path(baselinePickleFile)

    baselinePlusPropSigPickleFile = "experimentOutputs/jrule/2021-04-16T01:38:48.194736/jrule_arity=3_BO=False_CO=False_dp=False_doshaping=False_ES=1_ET=600_HR=1.0_it=1_MF=10_parallelTest=False_RS=5000_RT=3600_RR=False_RW=False_st=False_STM=True_TRR=default_K=2_topkNotMAP=False_tset=S12_DSL=False.pickle"
    baselinePlusPropSigResult, resumeGrammar, _ = resume_from_path(baselinePlusPropSigPickleFile)

    propSigOnlyPickleFile = "experimentOutputs/jrule/2021-04-16T01:33:54.134337/jrule_arity=3_BO=False_CO=False_dp=False_doshaping=False_ES=1_ET=600_HR=1.0_it=1_MF=10_parallelTest=False_RS=5000_RT=3600_RR=False_RW=False_st=False_STM=True_TRR=default_K=2_topkNotMAP=False_tset=S12_DSL=False.pickle"
    propSigOnlyResult, resumeGrammar, _ = resume_from_path(propSigOnlyPickleFile)

    recognizerNames = ['learned', 'combined', 'prop_sig']
    # evaluateRecognizers(resumeGrammar, [baselineResult, baselinePlusPropSigResult, propSigOnlyResult], recognizerNames)
    # evaluateRecognizersForTask(resumeGrammar, [baselineResult, baselinePlusPropSigResult, propSigOnlyResult], recognizerNames, taskToInvestigate="058_1", request=tlist(t0), productionToInvestigate="cdr", printGrammar=True)

    # print(propSigOnlyResult.recognitionModel.generativeModel)

    request = arrow(tlist(tint), tlist(tint))

    numSamples = 100

    samples = propSigOnlyResult.recognitionModel.sampleManyHelmholtz(requests=[request], N=numSamples, CPUs=1)
    for frontier in samples:
        print("Task:{}\nProgram: {}\n".format("\n".join(["{} -> {}".format(i[0],o) for i,o in frontier.task.examples]), frontier.entries[0].program))


    # prims = bootstrapTarget_extra(useInts=True)
    # grammar = Grammar.uniform(prims)
    # print(grammar)
    # request = arrow(tlist(tint), tlist(tint))
    # print("Sampling with request: {}".format(request))

    # for i in range(numSamples):
    #     sample = grammar.sample(request=request)
    #     print("Program: {}".format(sample))

    # print("--------------------------------------------------------------------------------------------------")

    # toutputToList = Primitive("tlist_to_toutput", arrow(tlist(tint), toutput), lambda x: x)
    # tinputToList = Primitive("tinput_to_tlist", arrow(tinput, tlist(tint)), lambda x: x)
    # prims = prims + [toutputToList, tinputToList]
    # grammar = Grammar.fromProductions([(3.0, p) if p.name == "tinput_to_tlist" else (1.0, p) for p in prims])
    # print(grammar)
    # request = arrow(tinput, toutput)
    # print("Sampling with request: {}".format(request))

    # for i in range(numSamples):
    #     sample = grammar.sample(request=request)
    #     print("Program: {}".format(sample))

    return





