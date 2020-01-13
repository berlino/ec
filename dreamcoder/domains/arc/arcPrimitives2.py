from dreamcoder.program import Primitive, Program, baseType
from dreamcoder.grammar import Grammar
from dreamcoder.type import tlist, tint, tbool, arrow, t0, t1, t2

from functools import reduce
import json
import numpy as np


_black = 0
_blue = 1
_red = 2
_green = 3
_yellow = 4
_grey = 5
_pink = 6
_orange = 7
_teal = 8
_maroon = 9

def _if(c): return lambda t: lambda f: t if c else f

def _replace(x): return lambda t: lambda f: lambda flag: _if(flag)(t)(f)
def _eq(x): return lambda y: x == y
def _add(x): return lambda y: x + y
def _expand(x): return lambda count: [x for _ in range(count)]

def _flatten(l): return [x for xs in l for x in xs]
def _concat(l): return lambda y: l + y
def _zip(l): return lambda b: lambda f: list(map(lambda x,y: f(x)(y), l, b))

def _concat2d(l): return lambda b: _zip(l)(b)(_concat)
def _3dTo2d(l): return _reduce(lambda a: lambda b: _concat2d(a)(b))([[],[],[]])(l)
# def _cutRows(l): return lambda start: lambda end: _concat(_keepRows(l)(0)(start))(_keepRows(l)(end)(len(l)))
def _keepCols(l): return lambda start: lambda end: _map(lambda row: _filteri(lambda i: lambda x: i >= start and i < end)(row))(l)
def _keepRows(l): return lambda start: lambda end: l[start:end]
def _elementwiseAdd(l): return lambda b: _mapi(lambda i: lambda x: _zip(l[i])(b[i])(_add))(l)

def _replaceColor(l): return lambda c: lambda new: _map(lambda m: _map(lambda x: _replace(x)(new)(x)(_eq(x)(c)))(m))(l)
def _findNonBlackColor(l): return _filter(lambda x: not _eq(0)(x))(_flatten(l))[0]
# def _horizontalOverlap(l): return lambda b: lambda numRows: _concat(_keepRows(l)(0)(len(l)-numRows))(_keepRows(b)(numRows)(len(b)))
def _replaceNonBlackColor(l): return lambda new: _replaceColor(l)(_findNonBlackColor(l))(new)


def _solve1(l):
    temp = _map(lambda m: _map(lambda current: _replace(current)(_expand(_expand(0)(3))(3))(l)(_eq(current)(0)))(m))(l)
    return _flatten(_map(lambda blockRow: _3dTo2d(blockRow))(temp))
def _solve6(l): return _replaceNonBlackColor(_elementwiseAdd(_keepCols(l)(0)(3))(_keepCols(l)(4)(7)))(0)
# def _solve6(l): return _replaceNonBlackColor(_elementwiseAdd(_keepCols(l)(0)(3))(_keepCols(l)(4)(7)))(0)

def _reduce(f): return lambda x0: lambda l: reduce(lambda a, x: f(a)(x), l, x0)
def _map(f): return lambda l: list(map(f, l))
def _mapi(f): return lambda l: list(map(lambda i_x: f(i_x[0])(i_x[1]), enumerate(l)))
def _filter(f): return lambda l: list(filter(f, l))
def _filteri(f): return lambda l: list(_map(lambda x: x[1])(filter(lambda i_x: f(i_x[0])(i_x[1]), enumerate(l))))


class RecursionDepthExceeded(Exception):
    pass


tcolor = baseType('color')
tmask = tlist(tlist(tcolor))

def basePrimitives():


    return [

    Primitive("black", tcolor, _black),
    Primitive("blue", tcolor, _blue),
    Primitive("red", tcolor, _red),
    Primitive("green", tcolor, _green),
    Primitive("yellow", tcolor, _yellow),
    Primitive("grey", tcolor, _grey),
    Primitive("pink", tcolor, _pink),
    Primitive("orange", tcolor, _orange),
    Primitive("teal", tcolor, _teal),
    Primitive("maroon", tcolor, _maroon),


    Primitive("if", arrow(tbool, t0, t0, t0), _if),

    Primitive("replace", arrow(arrow(tcolor, t0, tbool), tlist(t0), tlist(t0), tlist(t0)), _replace),
    Primitive("eq", arrow(tcolor, tcolor, tbool), _eq),
    Primitive("add", arrow(tcolor, tcolor, tcolor), _add),
    Primitive("expand", arrow(t0, tcolor, tlist(t0)), _expand),

    Primitive("flatten", arrow(tlist(tlist(t0)), tlist(t0)), _flatten),
    Primitive("concat", arrow(tlist(t0), tlist(t0), tlist(t0)), _concat),
    Primitive("zip", arrow(tlist(t0), tlist(t1), arrow(t0, t1, t2), tlist(t2)), _zip),

    Primitive("concat2d", arrow(tmask, tmask, tmask), _concat2d),
    Primitive("3dTo2d", arrow(tlist(tmask), tmask), _3dTo2d),
    # Primitive('cutRows', arrow(tmask, tint, tint, tmask), _cutRows),
    Primitive('keepCols', arrow(tmask, tint, tint, tmask), _keepCols),
    Primitive('keepRows', arrow(tmask, tint, tint, tmask), _keepRows),
    Primitive('elementwiseAdd', arrow(tmask, tmask, tmask), _elementwiseAdd),

    Primitive('replaceColor', arrow(tmask, tcolor, tcolor, tmask), _replaceColor),
    Primitive('findNonBlackColor', arrow(tmask, tcolor), _findNonBlackColor),
    Primitive('replaceNonBlackColor', arrow(tmask, tcolor, tmask), _replaceNonBlackColor),

    Primitive("solve1", arrow(tmask, tmask), _solve1),
    # Primitive("solve6", arrow(tmask, tmask), _solve6),

    Primitive("reduce", arrow(arrow(t1, t0, t1), t1, tlist(t0), t1), _reduce),
    Primitive("map", arrow(arrow(t0, t1), tlist(t0), tlist(t1)), _map),
    Primitive("mapi",arrow(arrow(tint,t0,t1), tlist(t0), tlist(t1)),_mapi),
    Primitive("filter", arrow(arrow(t0, tbool), tlist(t0), tlist(t0)), _filter),
    Primitive("filteri", arrow(arrow(tint, t0, t1), tlist(t0), tlist(t1)), _filteri)]


def retrieveARCJSONTask(filename, directory):
    with open(directory + '/' + filename, "r") as f:
        loaded = json.load(f)

    train = Task(filename, arrow(tmask, tmask), [((example['input'],), example['output']) for example in loaded['train']])
    test = Task(filename, arrow(tmask, tmask), [((example['input'],), example['output']) for example in loaded['test']])

    return train, test

def pprint(arr):
    print('The grid shape is {}'.format(np.array(arr).shape))
    for row in arr:
        print(row)
#
if __name__ == "__main__":
#     input = [[7, 0, 7], [7, 0, 7], [7, 7, 0], [2,2,2], [2,2,2], [1,1,1]]
#     output = [[7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 0, 7, 0, 0, 0, 7, 0, 7],
#                                                   [7, 7, 0, 0, 0, 0, 7, 7, 0], [7, 0, 7, 0, 0, 0, 7, 0, 7],
#                                                   [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 7, 0, 0, 0, 0, 7, 7, 0],
#                                                   [7, 0, 7, 7, 0, 7, 0, 0, 0], [7, 0, 7, 7, 0, 7, 0, 0, 0],
#                                                   [7, 7, 0, 7, 7, 0, 0, 0, 0]]
#
#     a,b,c = [[1,1,1],[2,2,2],[3,3,3]],[[2,2,2],[3,3,3],[4,4,4]], [[3,3,3],[4,4,4],[5,5,5]]
#
    input6 = [[1, 0, 0, 5, 0, 1, 0], [0, 1, 0, 5, 1, 1, 1], [1, 0, 0, 5, 0, 0, 0]]

    pprint(_solve6(input6))
#     #
    # print(_zip(a)(b)(lambda x: lambda y: x + y))

    # print(_transpose(np.zeros((3,3)).tolist()))
    # print(_flatten(_map(temp)(lambda blockRow: _reduce(lambda a: lambda b: [a[i] + b[i] for i in range(len(a))])([[],[],[]])(blockRow))))

    # def a(current): return _replace(current)(x)(_equals(current)(0))
    #
    # print(a(1))
