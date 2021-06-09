"""
A program with a name, a type and potentially cached values of its evaluation on Task objects
"""

from dreamcoder.type import arrow, tbool

class Property:
    
    def __init__(self, program, request, name=None, logPrior=None):
        self.program = program
        self.request = request
        self.name = name if name is not None else str(program)
        # only applicable if property was sampled from Grammar
        self.logPrior = logPrior
        self.cachedTaskEvaluations = {}

    def __str__(self):
        return self.name

    def updateCachedTaskEvaluations(self, task, value):
        key = task.describe()
        if key in self.cachedTaskEvaluations and (self.cachedTaskEvaluations[key] != value):
            raise Exception("Cached value for {},{} differs from new one (cached: {}, new one: {})".format(task, self.name, self.cachedTaskEvaluations[key], value))
        else:
            self.cachedTaskEvaluations[key] = value
        return

    def getValue(self, task):
        key = task.describe()
        if key in self.cachedTaskEvaluations:
            taskValue = self.cachedTaskEvaluations[key]
        else:
            taskValue = getTaskPropertyValue(self.program, task)
            self.cachedTaskEvaluations[key] = taskValue
        return taskValue

    def getTaskSignature(self, tasks):
        return [self.getValue(task) for task in tasks]

def getTaskPropertyValue(f, task):
    """
    Args:
        f (function): the property function
        task (Task): one of the tasks we are interesed in solving
    """

    taskPropertyValue = "mixed"
    try:
        exampleValues = [task.predict(f,[x[0], y]) for x,y in task.examples]
        if all([value is False for value in exampleValues]):
            taskPropertyValue = "allFalse"
        elif all([value is True for value in exampleValues]):
            taskPropertyValue = "allTrue"

    except Exception as e:
        # print(str(e))
        # print("Task name: {}".format(task.name))
        # print("Task example input: {}".format(task.examples[0][0][0]))
        # print("Task example output: {}".format(task.examples[0][0][1]))
        # print("Program: {}".format(program))
        taskPropertyValue = "mixed"

    return taskPropertyValue



