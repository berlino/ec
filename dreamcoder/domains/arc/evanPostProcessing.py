import pandas as pd

from dreamcoder.domains.arc.arcPrimitives import *
from dreamcoder.domains.arc.utilsPostProcessing import *
from dreamcoder.type import Context, arrow, tbool, tlist, tint, t0, UnificationFailure

def writeSyntheticNlData(paths, runNames):

    basePrimitives()
    leafPrimitives()

    with open("data/arc/language/human/train/language.json") as languageJson:
        humanNlPrograms = json.load(languageJson)
        humanNlPrograms = {str(key):value for key,value in humanNlPrograms.items()}
    toReturn = {}

    for path, runName in zip(paths, runNames):
        result, lastGrammar, firstGrammar = resume_from_path(path)

        for task in result.frontiersOverTime.keys():

            if task not in toReturn:
                toReturn[task] = {}

            manuallySolved = task.name in list(manuallySolvedTasks.keys())
            humanSolution = None
            if manuallySolved:
                humanProgram = parseHandwritten(task.name)
                # usesIdx = getProgramUses(humanProgram, arrow(tgridin, tgridout), firstGrammar)
                # humanSolution = (len(usesIdx), usesIdx)
                if "handwritten" not in toReturn[task]:
                    toReturn[task]["handwritten"] = humanProgram

            for generation, frontier in enumerate(result.frontiersOverTime[task]):
                frontier = frontier.topK(10)
                if len(frontier.entries) > 0:
                    for i,entry in enumerate(frontier.entries):
                        toReturn[task]["dc_{}_generation_{}_entry_{}".format(runName, generation, i)] = entry.program

    count = 0
    for task in toReturn.keys():
        print(task)
        if len(toReturn[task].keys()):
            count += 1

    # create multindex dataframe of size (# unique programs, 2)
    data, index = [], []
    for task, programDict in toReturn.items():
        for programInfo, program in programDict.items():
            rowData = programInfo.split("_")
            if len(rowData) > 1:
                row = [program, rowData[1], rowData[3], rowData[5], False]
            else:
                # if it is the humanProgram
                row = [program, None, None, None, True]
            row += humanNlPrograms.get(str(task), [])
            data.append(row)
            index.append(task)


    maxNlDescriptions = max([len(v) for v in humanNlPrograms.values()])
    columns = ["program", "DSL", "generation", "frontierEntryRank", "isHandWritten"] + ["nl_{}".format(i) for i in range(maxNlDescriptions-1)]
    df = pd.DataFrame(data, index=index, columns=columns)
    print("DF shape before dropping duplicates: {}".format(df.shape))
    df = df.drop_duplicates(subset=["program"])
    print("DF shape after dropping duplicates: {}".format(df.shape))
    # df.to_csv("data/arc/all_unique_programs.csv")
    return df

def main():

    # run that solved the most tasks ever with rich DSL and 6 iterations
    path = "experimentOutputs/arc/2020-05-10T14:49:21.186479/"
    pickleFilename = "arc_aic=1.0_arity=3_BO=True_CO=True_ES=1_ET=1200_t_zero=28800_HR=0.0_it=6_MF=10_noConsolidation=False_pc=1.0_RT=1800_RR=False_RW=False_solver=ocaml_STM=True_L=1.0_TRR=unsolved_K=2_topkNotMAP=False.pickle"
    picklePath1 = path + pickleFilename

    # run with rich DSL and 20 iterations and batching
    path = "experimentOutputs/arc/2020-05-01T19:00:26.769291/"
    pickleFilename = "arc_aic=1.0_arity=3_BO=True_CO=True_ES=1_ET=1200_t_zero=3600_HR=0.0_it=20_MF=10_noConsolidation=False_pc=1.0_RT=1200_RR=False_RW=False_solver=ocaml_STM=True_L=1.0_batch=30_TRR=randomShuffle_K=2_topkNotMAP=False.pickle"
    picklePath2 = path + pickleFilename

    # recent run of just iterative bigram fitting
    path = "experimentOutputs/arc/2021-04-23T16:33:13.110429/"
    pickleFilename = "arc_arity=3_BO=True_CO=True_ES=1_ET=1200_t_zero=3600_HR=0.0_it=9_MF=5_noConsolidation=False_RS=10000_RT=3600_RR=False_RW=False_solver=ocaml_STM=True_TRR=unsolved_K=2_topkNotMAP=False_DSL=False.pickle"
    picklePath3 = path + pickleFilename

    # recent dc run with trimmed DSL + arcCNN
    path = "experimentOutputs/arc/2021-04-24T20:09:56.975531/"
    pickleFilename = "arc_aic=1.0_arity=3_BO=True_CO=True_ES=1_ET=1200_t_zero=1_HR=0.0_it=9_MF=5_noConsolidation=False_pc=30.0_RS=10000_RT=3600_RR=False_RW=False_solver=ocaml_STM=True_L=1.5_TRR=unsolved_K=2_topkNotMAP=False_UET=14400.pickle"
    picklePath4 = path + pickleFilename

    writeSyntheticNlData([picklePath1, picklePath2, picklePath3, picklePath4], ["rich", "rich_2", "trimmed", "trimmed_2"])
