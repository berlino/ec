
# runFull = False
runFull = True

if runFull:
    from dreamcoder.program import Primitive, Program, baseType
    from dreamcoder.grammar import Grammar
    from dreamcoder.type import tlist, tint, tbool, arrow, t0, t1, t2
    #
from functools import reduce
import json
import numpy as np


# def _solve(input):
#
#     input = np.array(input)
#     d = np.array(input).shape[0] * 3
#     output = np.zeros((d,d))
#
#     nonZeroNum = input[input != 0][0]
#     ixs = 3 * np.argwhere(input == nonZeroNum)
#     for x,y in ixs:
#         output[x:x+3, y:y+3] = input
#     return output.tolist()

# class Grid:
#     def __init__(self, grid: list):
#         self.numRows = len(grid)
#         self.numCols = len(grid[0])
#         self.grid = np.array(grid)


class Object:
    def __init__(self, mask=[], points=None, isGrid=True):

        if points is None:
            self.points = {}
            self.numRows, self.numCols = len(mask), len(mask[0])
            for y in range(len(mask)):
                for x in range(len(mask[0])):
                    if not isGrid:
                        if mask[y][x] != 0:
                            self.points[(y,x)] = mask[y][x]
                    else:
                        self.points[(y, x)] = mask[y][x]

        else:
            self.points = points
            self.numRows = self.getNumRows()
            self.numCols = self.getNumCols()



    def getNumRows(self):
        return max([key[0] for key in self.points.keys()]) + 1

    def getNumCols(self):
        return max([key[1] for key in self.points.keys()]) + 1

    def pprint(self):
        temp = np.zeros((self.numRows,self.numCols))
        for yPos,xPos in self.points:
            temp[yPos, xPos] = self.points[(yPos,xPos)]
        pprint(temp)
        return temp.tolist()


    def move(self, y, x):
        newPoints = {}
        for yPos,xPos in self.points.keys():
            color = self.points[(yPos,xPos)]
            newPoints[(yPos + y, xPos + x)] = color
        return Object(points=newPoints)

    def grow(self, c):
        newPoints = {}
        for yPos, xPos in self.points.keys():
            color = self.points[(yPos, xPos)]
            newPoints[(c * yPos, c * xPos)] = color
            for kX in range(c):
                for kY in range(c):
                    newPoints[(c * yPos + kY, c * xPos + kX)] = color
        return Object(points=newPoints)


    def merge(self, object):
        newPoints = {key:value for key,value in self.points.items()}
        for yPos, xPos in object.points.keys():
            assert (yPos, xPos) not in newPoints
            newPoints[(yPos, xPos)] = object.points[(yPos, xPos)]
        return Object(points=newPoints)

    def split(self, isHorizontal, keepFirst):
        if isHorizontal:
            halfway = self.getNumRows() // 2
            topHalf = Object(points={(y,x): self.points[(y,x)] for y,x in self.points.keys() if y < halfway})
            if self.getNumRows() % 2 == 1:
                halfway += 1
            bottomHalf = Object(points={(y-halfway,x): self.points[(y,x)] for y,x in self.points.keys() if y >= halfway})
            return topHalf if keepFirst else bottomHalf
        else:
            halfway = self.getNumCols() // 2
            leftHalf = Object(points={(y,x): self.points[(y,x)] for y,x in self.points.keys() if x < halfway})
            if self.getNumCols() % 2 == 1:
                halfway += 1
            rightHalf = Object(points={(y,x-halfway): self.points[(y,x)] for y,x in self.points.keys() if x >= halfway})
            return leftHalf if keepFirst else rightHalf


    def concat(self, object, direction):

        if direction == 'right':
            return self.merge(object.move(0, self.numCols))
        elif direction == 'down':
            return self.merge(object.move(self.numRows, 0))

        elif direction == 'left':
            return object.merge(self.move(0, object.numCols))
        elif direction == 'up':
            return object.merge(self.move(object.numRows, 0))

        else:
            raise NotImplementedError

    def zipGrids(self, object, f):
        assert object.numCols == self.numCols and object.numRows == self.numRows
        assert set(object.points.keys()) == set(self.points.keys())
        return Object(points={key:f(self.points[key])(object.points[key]) for key in self.points.keys()})

    def __len__(self):
        return 100 * self.numRows + self.numCols

    def __eq__(self, other):
        if isinstance(other, Object):
            sameKeys = set(other.points.keys()) == set(self.points.keys())
            if sameKeys:
                allPointsEqual = all([self.points[key] == other.points[key] for key in self.points.keys()])
                return self.numRows == other.numRows and self.numCols == other.numCols and sameKeys and allPointsEqual
        return False

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


def _keepNonBlacks(c): return lambda c2: c2 if c != 0 else 0
def _replaceOverlapping(c): return lambda c2: _red if (c == c2 and c != _black) else _black

def _split(a): return lambda isHorizontal: lambda keepFirst: a.split(isHorizontal, keepFirst)
def _move(a): return lambda y: lambda x: a.move(y,x)
def _grow(a): return lambda n: a.grow(n)
def _concatN(a): return lambda b: lambda dir: lambda n: a.concat(b, dir) if n == 1 else _concatN(a.concat(b, dir))(b)(dir)(n-1)
def _duplicateN(a): return lambda dir: lambda n: _concatN(a)(a)(dir)(n)
def _zipGrids(a): return lambda b: lambda f: a.zipGrids(b, f)
def _solve1(a): return _zipGrids(_grow(a)(3))(_duplicateN(_duplicateN(a)('right')(2))('down')(2))(_keepNonBlacks)
def _solve6(a): return _zipGrids(_split(a)(False)(True))(_split(a)(False)(False))(_replaceOverlapping)


class RecursionDepthExceeded(Exception):
    pass

if runFull:
    tdirection = baseType('direction')
    tcolor = baseType('color')
    tgrid = baseType('tgrid')

def basePrimitives():


    return [
    #
    Primitive('0', tint, 0),
    Primitive('1', tint, 1),
    Primitive('2', tint, 2),
    Primitive('3', tint, 3),
    Primitive('4', tint, 4),
    Primitive('5', tint, 5),
    Primitive('6', tint, 6),
    Primitive('7', tint, 7),
    Primitive('8', tint, 8),

    Primitive('true', tbool, True),
    Primitive('false', tbool, False),

    Primitive("left", tdirection, "left"),
    Primitive("right", tdirection, "right"),
    Primitive("down", tdirection, "down"),
    Primitive("up", tdirection, "up"),

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

    Primitive('split', arrow(tgrid, tbool, tbool, tgrid), _split),
    Primitive('move', arrow(tgrid, tint, tint, tgrid), _move),
    Primitive('keepNonBlacks', arrow(tcolor, tcolor, tcolor), _keepNonBlacks),
    Primitive('grow', arrow(tgrid, tint, tgrid), _grow),
    Primitive('concatN', arrow(tgrid, tgrid, tdirection, tint, tgrid), _concatN),
    Primitive('duplicateN', arrow(tgrid, tdirection, tint, tgrid), _duplicateN),
    Primitive('zipGrids', arrow(tgrid, tgrid, arrow(tcolor, tcolor, tcolor), tgrid), _zipGrids),
    Primitive('solve6', arrow(tgrid, tgrid), _solve6),
    Primitive('solve1', arrow(tgrid, tgrid), _solve1)]


# def retrieveARCJSONTask(filename, directory):
#     with open(directory + '/' + filename, "r") as f:
#         loaded = json.load(f)
#
#     train = Task(filename, arrow(tgrid, tgrid), [(Object(mask=(example['input'],)), Object(mask=example['output'])) for example in loaded['train']])
#     test = Task(filename, arrow(tgrid, tgrid), [(Object(mask=(example['input'],)), Object(mask=example['output'])) for example in loaded['test']])
#
#     return train, test

def pprint(arr):
    print('The grid shape is {}'.format(np.array(arr).shape))
    for row in arr:
        print(row)
#
#     input = [[7, 0, 7], [7, 0, 7], [7, 7, 0], [2,2,2], [2,2,2], [1,1,1]]
#     output = [[7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 0, 7, 0, 0, 0, 7, 0, 7],
#                                                   [7, 7, 0, 0, 0, 0, 7, 7, 0], [7, 0, 7, 0, 0, 0, 7, 0, 7],
#                                                   [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 7, 0, 0, 0, 0, 7, 7, 0],
#                                                   [7, 0, 7, 7, 0, 7, 0, 0, 0], [7, 0, 7, 7, 0, 7, 0, 0, 0],
#                                                   [7, 7, 0, 7, 7, 0, 0, 0, 0]]
#
#     a,b,c = [[1,1,1],[2,2,2],[3,3,3]],[[2,2,2],[3,3,3],[4,4,4]], [[3,3,3],[4,4,4],[5,5,5]]
#


    # print(_zip(a)(b)(lambda x: lambda y: x + y))

    # print(_transpose(np.zeros((3,3)).tolist()))
    # print(_flatten(_map(temp)(lambda blockRow: _reduce(lambda a: lambda b: [a[i] + b[i] for i in range(len(a))])([[],[],[]])(blockRow))))

    # def a(current): return _replace(current)(x)(_equals(current)(0))
    #
    # print(a(1))

if __name__ == "__main__":

    input6 = [[1, 0, 0, 5, 0, 1, 0], [0, 1, 0, 5, 1, 1, 1], [1, 0, 0, 5, 0, 0, 0]]
    input7 = [[0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 5, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]

    input1 = [[0, 7, 7], [7, 7, 7], [0, 7, 7]]
    input2 = [[0, 0, 0], [0, 0, 0], [3, 3, 0]]

    a = Object(input6)
    b = Object(input2)

    a.pprint()
    res = _solve6(a)
    res.pprint()