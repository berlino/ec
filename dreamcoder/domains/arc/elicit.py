import numpy as np
import json
import pandas as pd
pd.set_option('display.max_colwidth', -1)
import pickle

from dreamcoder.domains.arc.utilsPostProcessing import getProgramUses
from dreamcoder.domains.arc.arcPrimitives import tgridin, tgridout
from dreamcoder.type import arrow

def parseNaturalLanguageAnnotations():

    # load dictionary that maps 0,1,..400 ids to Chollet ids
    id_to_name = pd.read_csv("data/arc/id_to_name.csv", names=["idx", "taskName"], dtype={"idx":np.int, "taskName":str}, header=0).to_dict()["taskName"]
    name_to_id = {name: int(idx) for idx,name in id_to_name.items()}

    nl = pd.read_csv("data/arc/successful_phrases.csv")
    nl["task_name"] = nl["task_number"].apply(lambda idx: id_to_name[idx])

    def concat(df):
        phrase = []
        for row in df.iterrows():
            phrase.append(row[1][0])
        return "||".join(phrase)

    temp = nl.set_index(["task_name", "phrase_kind", "nth_phrase_in_paragraph"]).loc[:, ["phrase"]]
    groupedNlDf = temp.groupby(level=[0,1]).apply(lambda df: concat(df)).rename("natural_language").to_frame()
    groupedNlDf = groupedNlDf.reset_index().set_index("task_name")
    return groupedNlDf

def getTaskToBestProgramUses(grammar, frontiers):
    taskToBestProgramUses = {}
    for task, frontier in frontiers.items():
        if len(frontier.entries) > 0:
            program = frontier.topK(1).entries[0].program
            taskToBestProgramUses[task] = getProgramUses(program, arrow(tgridin, tgridout), grammar)
        else:
            taskToBestProgramUses[task] = []
    return taskToBestProgramUses

def getPrimitiveUseCounts(grammar, taskToBestProgramUses):
    primitiveUseCounts = {primitive.name:0 for primitive in grammar.primitives}
    for task, uses in taskToBestProgramUses.items():
        for i in range(len(grammar.primitives)):
            if i in uses:
                primitiveUseCounts[grammar.primitives[i].name] += 1
        # print(task, [namesToDescriptions[grammar.primitives[idx].name][0] for idx in uses])
    return primitiveUseCounts

def createElicitBinaryClassificationTable(nlDf, grammar, frontiers, primitiveIdx):
    return


def main(grammar):

    # load dictionary from task to a list of the idxs of the primitives in the best program
    frontiers = pickle.load(open("data/arc/prior_enumeration_frontiers.pkl",  "rb"))
    taskToBestProgramUses = getTaskToBestProgramUses(grammar, frontiers)

    # load alternative names for primitives
    namesToDescriptions = json.load(open("data/arc/primitiveNamesToDescriptions.json", "r"))
    
    # find good candidate primitives to experiment with
    primitiveUseCounts = getPrimitiveUseCounts(grammar, taskToBestProgramUses)
    for primitive, numOfProgramUsed in sorted(primitiveUseCounts.items(), key=lambda tuple: tuple[1], reverse=True):
        print("Primitive: {} used in {} out of {} programs".format(primitive, numOfProgramUsed, len([f for f in frontiers.values() if len(f.entries) > 0])))

    # load nl annotations DataFrame
    nlDf = parseNaturalLanguageAnnotations()
    print(nlDf)

    return

