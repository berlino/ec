import json
import os
import subprocess

from pathos.multiprocessing import Pool

from dreamcoder.domains.arithmetic.arithmeticPrimitives import k1, k0, addition, subtraction, multiplication
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.grammar import Grammar
from dreamcoder.program import Program
from dreamcoder.recognition import RecognitionModel, DummyFeatureExtractor
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint
from dreamcoder.utilities import tuplify, timing, eprint, get_root_dir, mean


def helmholtzEnumeration(g, request, inputs, timeout, _=None,
                         special=None, evaluationTimeout=None):
    """Returns json (as text)"""
    message = {"request": request.json(),
               "timeout": timeout,
               "DSL": g.json(),
               "extras": [inputGrid.toJson() for inputGrid in inputs]}


    if evaluationTimeout: message["evaluationTimeout"] = evaluationTimeout
    if special: message["special"] = special
    message = json.dumps(message)
    with open('/tmp/hm', 'w') as handle:
        handle.write(message)
    try:
        binary = os.path.join(get_root_dir(), 'helmholtz')
        print("starting process")
        process = subprocess.Popen(binary,
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE)
        print("process opened")
        print("sending message: {}".format(message))
        response, error = process.communicate(bytes(message, encoding="utf-8"))
        print('response: {}'.format(response))
        print('error: {}'.format(error))
    except OSError as exc:
        raise exc
    return response


def backgroundHelmholtzEnumeration(tasks, g, timeout, _=None,
                                   special=None, evaluationTimeout=None):
    requests = list({t.request for t in tasks})

    # get unique inputs for each type
    inputs = {}
    for r in requests:
        inputs[r] = []
        inputHashes = set()
        for t in tasks:
            if t.request == r:
                for xs, y in t.examples:
                    print(xs)
                    if xs[0].hash() not in inputHashes:
                        inputs[r].append(xs[0])

    workers = Pool(len(requests))
    promises = [workers.apply_async(helmholtzEnumeration,
                                    args=(g, r, inputs[r], float(timeout)),
                                    kwds={'special': special,
                                          'evaluationTimeout': evaluationTimeout})
                for r in requests]

    def get():
        results = []
        for p in promises:
            print("calling get")
            res = p.get()
            print("called get")
            results.append(res)
        print("results", results)
        frontiers = []
        with timing("(Helmholtz enumeration) Decoded json into frontiers"):
            for request, result in zip(requests, results):
                response = json.loads(result.decode("utf-8"))
                for b, entry in enumerate(response):
                    frontiers.append(Frontier([FrontierEntry(program=Program.parse(p),
                                                             logPrior=entry["ll"],
                                                             logLikelihood=0.)
                                               for p in entry["programs"]],
                                              task=Task(str(b),
                                                        request,
                                                        [])))
        eprint("Total number of Helmholtz frontiers:", len(frontiers))
        return frontiers

    return get


if __name__ == "__main__":
    g = Grammar.uniform([k1, k0, addition, subtraction, multiplication])
    frontiers = helmholtzEnumeration(g,
                                     arrow(tint, tint),
                                     [[0], [1], [2]],
                                     10.)
    eprint("average frontier size", mean(len(f.entries) for f in frontiers))
    f = DummyFeatureExtractor([])
    r = RecognitionModel(f, g, hidden=[], contextual=True)
    r.trainBiasOptimal(frontiers, frontiers, steps=70)
    g = r.grammarOfTask(frontiers[0].task).untorch()
    frontiers = helmholtzEnumeration(g,
                                     arrow(tint, tint),
                                     [[0], [1], [2]],
                                     10.)
    for f in frontiers:
        eprint(f.summarizeFull())
    eprint("average frontier size", mean(len(f.entries) for f in frontiers))
