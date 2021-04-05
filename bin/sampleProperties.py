# experiments for Josh Rule
try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

import copy

from dreamcoder.dreaming import *
from dreamcoder.domains.list.main import retrieveJSONTasks
from dreamcoder.domains.list.makeListTasks import joshTasks, sortBootstrap
from dreamcoder.domains.list.listPrimitives import josh_primitives, bootstrapTarget_extra
from dreamcoder.domains.list.taskProperties import handWrittenProperties, tinput, toutput
from dreamcoder.enumeration import *
from dreamcoder.grammar import Grammar
from dreamcoder.type import *
from dreamcoder.utilities import tuplify

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("--w","-w",default="1",type=str)
    # parser.add_argument("--dream","-d",default=False,action='store_true')
    parser.add_argument("--timeout","-t",default=600,type=float)
    parser.add_argument("--CPUs",default=numberOfCPUs(),type=int)
    parser.add_argument("--solver",default="ocaml",type=str)
    parser.add_argument("--primitives", default="property_prims", choices=[
        "josh_1",
        "josh_2",
        "josh_3",
        "josh_3.1",
        "josh_final",
        "property_prims",
        "list_prims"])
    parser.add_argument(
        "--dataset",
        type=str,
        default="Lucas-old",
        choices=[
            "josh_1",
            "josh_2",
            "josh_3",
            "josh_3.1",
            "josh_final",
            "Lucas-old"])


    arguments = parser.parse_args()

    primitives = {
        "josh_1": joshTasks("1"),
        "josh_2": joshTasks("2"),
        "josh_3": joshTasks("3"),
        "josh_3.1": joshTasks("3.1"),
        "josh_final": joshTasks("final"),
        "property_prims": handWrittenProperties() + [Primitive(str(j), tint, j) for j in range(10)],
        "list_prims": bootstrapTarget_extra()
    }

    tasks = {
        "Lucas-old": lambda: retrieveJSONTasks("data/list_tasks.json") + sortBootstrap(),
        "josh_1": lambda: joshTasks("1"),
        "josh_2": lambda: joshTasks("2"),
        "josh_3": lambda: joshTasks("3"),
        "josh_3.1": lambda: joshTasks("3.1"),
        "josh_final": lambda: joshTasks("final"),
    }[arguments.dataset]()
    

    if "josh" in arguments.dataset:
        tasks = [t for t in tasks if int(t.name[:3]) < 81 and "_1" in t.name]

    tasks = [t for t in tasks if (t.request == arrow(tlist(tint), tlist(tint)) and isinstance(t.examples[0][1],list))]
    print("{} tasks".format(len(tasks)))

    g = Grammar.uniform(primitives[arguments.primitives])
    request = arrow(tinput, toutput, tbool)

    # allNegativeTasks = []
    # tasksCopy = copy.deepcopy(tasks)

    for t in tasks:
        t.request = request
        tCopy = copy.deepcopy(t)
        t.examples = [(tuplify([io[0][0], io[1]]), True) for io in t.examples]

    #     tCopy.examples = [(tuplify([io[0][0], io[1]]), False) for io in tCopy.examples] + [(tuplify([io[0][0], io[1]]),True) for task in tasksCopy for io in task.examples]
    #     assert tCopy.examples[0][1] != t.examples[0][1]
    #     allNegativeTasks.append(tCopy)
    # tasks = tasks + allNegativeTasks

    # program = g.sample(request)
    # print(program)



    frontiers, times, pcs = multicoreEnumeration(g,tasks,solver=arguments.solver,maximumFrontier=10,
                                                 enumerationTimeout= arguments.timeout,CPUs=arguments.CPUs,
                                                 evaluationTimeout=0.01,
                                                 testing=True, allTasks=tasks)
    
    # properties = {}

    for frontier in frontiers:
        print("\n Task: {} ".format(frontier.task))
        topK = frontier.topK(3)
        for frontierEntry in topK:
            print("Program (score = {}): {}".format(frontierEntry.logLikelihood, frontierEntry.program))

    #     for entry in frontier.entries:
    #         tasks_with_property = properties.get(entry.program, set())
    #         if frontier.task.name not in tasks_with_property:
    #             tasks_with_property.add(frontier.task.name)
    #         properties[entry.program] = tasks_with_property
    
    # sorted_properties = [(property_program, len(properties[property_program])) for property_program in properties.keys()]
    # sorted_properties = sorted(sorted_properties, key=lambda x: x[1])

    # for program_property, task_count in sorted_properties:
    #     print("Program property: {} matched with {} task".format(program_property, task_count))
    #     if task_count == 1:
    #         print("The task is: {}".format(list(properties[program_property])[0]))