# runFull = False
runFull = True

if runFull:
    from dreamcoder.program import Primitive, Program, baseType
    from dreamcoder.grammar import Grammar
    from dreamcoder.type import tlist, tint, tbool, arrow, t0, t1, t2, tpair
    #
from functools import reduce
import json
import numpy as np
import os

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

class Block:

    def __init__(self, points, originalGrid=None):
        self.points = points
        self.originalGrid = originalGrid

    def getMaxY(self):
        return max([key[0] for key in self.points.keys()])
    def getMaxX(self):
        return max([key[1] for key in self.points.keys()])
    def getMinY(self):
        return min([key[0] for key in self.points.keys()])
    def getMinX(self):
        return min([key[1] for key in self.points.keys()])

    def inGrid(self):
        return self.getMaxY() <= self.originalGrid.getMaxY() \
               and self.getMaxX() <= self.originalGrid.getMaxX() \
               and self.getMinY() >= self.originalGrid.getMinY() \
               and self.getMinX() >= self.originalGrid.getMinX()

    def fromPoints(self, points):
        return Block(points, originalGrid=self.originalGrid)

    def pprint(self, dims=None):
        # if dims is None:
        #     maxY, maxX = self.getMaxY(), self.getMaxX()
        # else:
        #     maxY, maxX= dims[0], dims[1]

        temp = self.toMinGrid(backgroundColor = '-')
        # temp = np.full((self.originalGrid.getNumRows(), self.originalGrid.getNumCols()), '-')
        # temp = np.full((maxY+1,maxX+1), '-')
        # for yPos,xPos in self.points:
        #     temp[yPos, xPos] = self.points[(yPos,xPos)]
        temp.pprint()
        return temp.tolist()

    def reflect(self, horizontal):
        res = {}
        for y,x in self.points.keys():
            if horizontal:
                res[((self.getMaxY()-y) + self.getMinY(), x)] = self.points[(y,x)]
            else:
                res[(y, (self.getMaxX()-x) + self.getMinX())] = self.points[(y,x)]

        return self.fromPoints(res)

    def move(self, y, x, keepOriginal=False):
        newPoints = self.points.copy() if keepOriginal else {}
        for yPos,xPos in self.points.keys():
            color = self.points[(yPos,xPos)]
            newPoints[yPos + y, xPos + x] = color
        return self.fromPoints(newPoints)

    def grow(self, c):
        newPoints = {}
        for yPos, xPos in self.points.keys():
            color = self.points[(yPos, xPos)]
            newPoints[(c * yPos, c * xPos)] = color
            for kX in range(c):
                for kY in range(c):
                    newPoints[(c * yPos + kY, c * xPos + kX)] = color

        return self.fromPoints(newPoints)

    def merge(self, object):
        # print(self.points)
        # print(object.points)
        newPoints = {key:value for key,value in self.points.items()}
        for yPos, xPos in object.points.keys():
            assert (yPos, xPos) not in newPoints
            newPoints[(yPos, xPos)] = object.points[(yPos, xPos)]
        return self.fromPoints(newPoints)

    def isRectangle(self, full=True):
        sortedPoints = sorted(self.points.keys())
        topLeftCorner = sortedPoints[0]
        bottomRightCorner = sortedPoints[-1]
        numCols = bottomRightCorner[1] - topLeftCorner[1] + 1
        numRows = bottomRightCorner[0] - topLeftCorner[0] + 1
        if numCols < 1 or numRows < 1:
            return False

        for x in range(topLeftCorner[1], topLeftCorner[1] + numCols):
            if not (topLeftCorner[0], x) in sortedPoints:
                return False
            if not (topLeftCorner[0] + numRows - 1, x) in sortedPoints:
                return False

        for y in range(topLeftCorner[0] + 1, topLeftCorner[0] + numRows - 1):
            if not (y, topLeftCorner[1]) in sortedPoints:
                return False
            if not (y, topLeftCorner[1] + numCols - 1) in sortedPoints:
                return False


        for y,x in sortedPoints:
            if not (y >= topLeftCorner[0] and y <= bottomRightCorner[0]):
                return False
            if not (x >= topLeftCorner[1] and x <= bottomRightCorner[1]):
                return False

        if full:
            if not len(sortedPoints) == numCols * numRows:
                return False
        return True

    def split(self, isHorizontal):
        if isHorizontal:
            horizontalLength = (self.getMaxY() - self.getMinY())
            halfway = horizontalLength  // 2
            topHalf = {(y,x): self.points[(y,x)] for y,x in self.points.keys() if y < halfway}
            if self.getNumRows() % 2 == 1:
                halfway += 1
            bottomHalf = {(y-halfway,x): self.points[(y,x)] for y,x in self.points.keys() if y >= halfway}
            return self.fromPoints(topHalf), self.fromPoints(bottomHalf)
        else:
            verticalLength = self.getMaxX() - self.getMinX()
            halfway = (self.getMaxX() - self.getMinX()) // 2
            leftHalf = {(y,x): self.points[(y,x)] for y,x in self.points.keys() if x < halfway}
            if verticalLength % 2 == 1:
                halfway += 1
            rightHalf = {(y,x-halfway): self.points[(y,x)] for y,x in self.points.keys() if x >= halfway}
            return self.fromPoints(leftHalf), self.fromPoints(rightHalf)

    def isSymmetrical(self, isHorizontal):
        firstHalf, secondHalf = self.split(isHorizontal)
        firstHalfReflect, secondHalfReflect = self.reflect(isHorizontal).split(isHorizontal)
        return firstHalf == firstHalfReflect and secondHalf == secondHalfReflect

    def splitBlockByColor(self, backgroundColor=None):
        blocks = []
        colors = set(self.points.values())
        if backgroundColor:
            colors.remove(backgroundColor)
        for c in colors:
            blocks.append(self.findBlockByColor(c))
        return blocks

    # def zipWithBlock(self, block, f):
    #     assert block.numCols == self.numCols and block.numRows == self.numRows
    #     assert set(block.points.keys()) == set(self.points.keys())
    #     newPoints = {key:f(self.points[key])(block.points[key]) for key in self.points.keys()}
    #     return self.fromPoints(newPoints)

    def toGrid(self, numRows, numCols, backgroundColor=0, withOriginal=False):
        grid = np.full((numRows, numCols), backgroundColor)
        for y in range(numRows):
            for x in range(numCols):
                if (y,x) in self.points:
                    grid[y,x] = self.points[(y,x)]
                else:
                    grid[y,x] = self.originalGrid.points[y,x] if withOriginal else backgroundColor
        return Grid(gridArray=grid)

    def __len__(self):
        return 100 * self.numRows + self.numCols

    def __eq__(self, other):
        if isinstance(other, Block):
            sameKeys = set(other.points.keys()) == set(self.points.keys())
            if sameKeys:
                allPointsEqual = all([self.points[key] == other.points[key] for key in self.points.keys()])
                return allPointsEqual
        return False

    def createEdgeMap(self, colors=[i for i in range(1,10)], isCorner=True):
        edges = {}
        for y,x in [key for key in self.points.keys()]:
            if self.points[(y,x)] in colors:
                edges[(y, x)] = []
                if isCorner:
                    adjacent = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
                else:
                    adjacent = [(-1, 0), (0, -1), (0, 1), (1, 0)]
                for y_inc, x_inc in adjacent:
                    if (y + y_inc, x + x_inc) in self.points and self.points[(y + y_inc, x + x_inc)] in colors:
                        edges[(y, x)] += [(y + y_inc, x + x_inc)]
        return edges

    def boxBlock(self):
        newPoints = self.points.copy()
        for y in range(self.getMinY(), self.getMaxY()+1):
            for x in range(self.getMinX(), self.getMaxX()+1):
                if (y,x) not in newPoints:
                    newPoints[(y,x)] = self.originalGrid.points[y,x]
        return self.fromPoints(newPoints)

    def fillPattern(self, color, maskFunc = lambda x: True):
        newPoints = self.points.copy()
        for key in self.points.keys():
            if maskFunc(key):
                newPoints[key] = color
        return self.fromPoints(newPoints)

    def replaceColors(self, cOld, cNew):
        newPoints = self.points.copy()
        for key, color in newPoints.items():
            if color == cOld:
                newPoints[key] = cNew
        return self.fromPoints(newPoints)

    def toMinGrid(self, backgroundColor=0, withOriginal=False):
        points = {(key[0] - self.getMinY(), key[1] - self.getMinX()): self.points[key] for key in self.points.keys()}
        grid = np.full((self.getMaxY() - self.getMinY() + 1, self.getMaxX() - self.getMinX() + 1), backgroundColor)
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if (y,x) in points:
                    grid[y,x] = points[(y,x)]
                else:
                    grid[y,x] = self.originalGrid.points[y+self.getMinY(),x+self.getMinX()] if withOriginal else backgroundColor
        return Grid(gridArray=grid)



class RectangleBlock(Block):

    def __init__(self, points, originalGrid=None):
        self.topLeftTile = min([point for point in points], key=lambda x: x[0] + x[1])
        super().__init__(points, originalGrid)
        # self.numRows, self.getNumCols = (self.points[-1][0] - self.topLeftTile[0] + 1), (self.points[-1][1] - self.topLeftTile[1] + 1)
    #     # for y in range(topLeftCorner[0], topLeftCorner[0] + numRows):
    #     #     for x in range(topLeftCorner[1], topLeftCorner[1] + numCols):
    #     #         points[(y,x)] =

    def getNumRows(self):
        return max([key[0] for key in self.points.keys()]) - min([key[0] for key in self.points.keys()]) + 1

    def getNumCols(self):
        return max([key[1] for key in self.points.keys()]) - min([key[1] for key in self.points.keys()]) + 1

    def fromPoints(self, points):
        return RectangleBlock(points, originalGrid=self.originalGrid)

    def split(self, isHorizontal):
        if isHorizontal:
            halfway = self.getNumRows() // 2
            topHalf = self.fromPoints(points={(y,x): self.points[(y,x)] for y,x in self.points.keys() if y < halfway})
            if self.getNumRows() % 2 == 1:
                halfway += 1
            bottomHalf = self.fromPoints(points={(y-halfway,x): self.points[(y,x)] for y,x in self.points.keys() if y >= halfway})
            return topHalf,bottomHalf
        else:
            halfway = self.getNumCols() // 2
            leftHalf = self.fromPoints(points={(y,x): self.points[(y,x)] for y,x in self.points.keys() if x < halfway})
            if self.getNumCols() % 2 == 1:
                halfway += 1
            rightHalf = self.fromPoints({(y,x-halfway): self.points[(y,x)] for y,x in self.points.keys() if x >= halfway})
            return leftHalf,rightHalf

    def concat(self, object, direction):

        if direction == 'right':
            temp = object.move(0, self.getNumCols())
            return self.merge(temp)
        elif direction == 'down':
            temp = object.move(self.getNumRows(), 0)
            return self.merge(temp)
        elif direction == 'left':
            return object.merge(self.move(0, object.getNumCols()))
        elif direction == 'up':
            return object.merge(self.move(object.getNumRows(), 0))

        else:
            raise NotImplementedError

    def concatUntilEdge(self, object, direction):
        block = self
        while block.concat(object, direction).inGrid():
            block = block.concat(object, direction)
        block = (block.concat(object, direction))
        return self.fromPoints({key:block.points[key] for key in block.points.keys() if self.originalGrid.contains(key)})

    def toMask(self):
        mask = self.fromPoints({key:True for key in self.points.keys()})
        return mask.fillIn(True)

    def fillIn(self, color):
        newPoints = self.points.copy()
        for y in range(self.topLeftTile[0] + 1, self.topLeftTile[0] + self.getNumRows() - 1):
            for x in range(self.topLeftTile[1] + 1, self.topLeftTile[1] + self.getNumCols() - 1):
                newPoints[(y,x)] = color
        return self.fromPoints(newPoints)

    # def pprint(self, dims=None):
    #     temp = np.full((self.topLeftTile[0] + self.getNumRows(), self.topLeftTile[1] + self.getNumCols()), '-')
    #     for yPos,xPos in self.points:
    #         temp[yPos, xPos] = self.points[(yPos,xPos)]
    #     pprint(temp)
    #     return temp.tolist()

class Grid(RectangleBlock):
    def __init__(self, gridArray=None, points=None, originalGrid=None):
        self.points = {}
        if points is not None:
            self.points = points
        else:
            for y in range(len(gridArray)):
                for x in range(len(gridArray[0])):
                    self.points[(y, x)] = gridArray[y][x]
        super().__init__(self.points, originalGrid)



    def fromPoints(self, points):
        return Grid(points=points, originalGrid=self.originalGrid)

    def contains(self, key):
        y,x = key
        return (y >= self.topLeftTile[0] and y < self.topLeftTile[0] + self.getNumRows()) \
               and (x >= self.topLeftTile[1] and x < self.topLeftTile[1] + self.getNumCols())

    def zipGrids(self, object, f):
        assert set(object.points.keys()) == set(self.points.keys())
        return Grid(points={key:f(self.points[key])(object.points[key]) for key in self.points.keys()})

    def pprint(self):
        temp = np.full((self.getNumRows(),self.getNumCols()),None)
        for yPos,xPos in self.points:
            temp[yPos, xPos] = self.points[(yPos,xPos)]
        pprint(temp)
        return temp.tolist()

    def findBlocksByColor(self, c, isCorner=True, boxBlocks=False):

        def dfs(v, edges, visited):
            visited.add(v)
            for adjacent in edges[v]:
                if adjacent not in visited:
                    dfs(adjacent, edges, visited)
            return

        visited = set()
        vertices = [v for v in self.points.keys() if self.points[v] == c]
        edges = self.createEdgeMap(colors=[c], isCorner=isCorner)
        blocks = []

        for v in vertices:
            if v not in visited:
                connected = set()
                dfs(v, edges, connected)
                blocks.append(connected)
                visited = visited.union(connected)

        if boxBlocks:
            blocks = [block.boxBlock() for block in blocks]

        return [Block(points={key:self.points[key] for key in block}, originalGrid=self) for block in blocks]

    def findSameColorBlocks(self, backgroundColor=None, isCorner=True, boxBlocks=False):
        blocks = []
        colors = set(self.points.values())
        if backgroundColor:
            colors.remove(backgroundColor)
        for c in colors:
            blocks.extend(self.findBlocksByColor(c, isCorner=isCorner))

        if boxBlocks:
            return [Block(points={key:self.points[key] for key in block.points.keys()}, originalGrid=self).boxBlock() for block in blocks]
        else:
            return [Block(points={key:self.points[key] for key in block.points.keys()}, originalGrid=self) for block in blocks]

    def findBlocksBy(self, backgroundColor=0, colors=[i for i in range(1,10)], isCorner=True, boxBlocks=False):

        if backgroundColor == 0 and backgroundColor in colors:
            colors.remove(backgroundColor)

        def dfs(v, edges, visited):

            visited.add(v)
            for adjacent in edges[v]:
                if adjacent not in visited:
                    dfs(adjacent, edges, visited)
            return

        visited = set()
        vertices = [v for v in self.points if self.points[v] in colors]
        edges = self.createEdgeMap(colors, isCorner)
        blocks = []

        for v in vertices:
            if v not in visited:
                connected = set()
                dfs(v, edges, connected)
                blocks.append(connected)
                visited = visited.union(connected)

        if boxBlocks:
            return [Block(points={key:self.points[key] for key in block}, originalGrid=self).boxBlock() for block in blocks]
        else:
            return [Block(points={key:self.points[key] for key in block}, originalGrid=self) for block in blocks]

    def findTouchingRectangles(self, backgroundColor, full=True):
        blocks = self.findBlocksBy(backgroundColor, isCorner=False)
        return [RectangleBlock(points=block.points.copy(), originalGrid=self) for block in blocks if block.isRectangle(full)]

    def findSameColorRectangles(self, backgroundColor=None, full=True):
        blocks = self.findSameColorBlocks(backgroundColor=backgroundColor)
        return [RectangleBlock(points=block.points.copy(), originalGrid=self) for block in blocks if block.isRectangle(full)]

    def colorAll(self, c, backgroundColor=None):
        if backgroundColor is None:
            backgroundColor = np.argsort(np.bincount(list(self.points.values())))[-1]
        return Grid(points={key:c if color != backgroundColor else backgroundColor for key,color in self.points.items()})

    def maskAndCenter(self, mask):
        return self.fromPoints(points={(key[0] - mask.topLeftTile[0], key[1] - mask.topLeftTile[1]):color for key,color in self.points.items() if key in mask.points})

#
#
# class Object:
#     def __init__(self, mask=[], points=None, backgroundColor=0, isGrid=True):
#
#         if points is None:
#             self.points = {}
#             self.numRows, self.numCols = len(mask), len(mask[0])
#             for y in range(len(mask)):
#                 for x in range(len(mask[0])):
#                     if not isGrid:
#                         if mask[y][x] != backgroundColor:
#                             self.points[(y,x)] = mask[y][x]
#                     else:
#                         self.points[(y, x)] = mask[y][x]
#
#         else:
#             self.points = points
#             self.numRows = self.getNumRows()
#             self.numCols = self.getNumCols()
#         self.backgroundColor = backgroundColor
#
#
#
#     def getNumRows(self):
#         return max([key[0] for key in self.points.keys()]) + 1
#
#     def getNumCols(self):
#         return max([key[1] for key in self.points.keys()]) + 1
#
#     def pprint(self):
#         temp = np.full((self.numRows,self.numCols), self.backgroundColor)
#         for yPos,xPos in self.points:
#             temp[yPos, xPos] = self.points[(yPos,xPos)]
#         pprint(temp)
#         return temp.tolist()
#
#     def reflect(self, horizontal):
#
#         res = {}
#         for y,x in self.points.keys():
#             if horizontal:
#                 res[(self.getNumRows()-1 - y, x)] = self.points[(y,x)]
#             else:
#                 res[(y, self.getNumCols()-1 - x)] = self.points[(y,x)]
#
#         return Object(points=res)
#
#     def move(self, y, x):
#         newPoints = {}
#         for yPos,xPos in self.points.keys():
#             color = self.points[(yPos,xPos)]
#             newPoints[(yPos + y, xPos + x)] = color
#         return Object(points=newPoints)
#
#     def grow(self, c):
#         newPoints = {}
#         for yPos, xPos in self.points.keys():
#             color = self.points[(yPos, xPos)]
#             newPoints[(c * yPos, c * xPos)] = color
#             for kX in range(c):
#                 for kY in range(c):
#                     newPoints[(c * yPos + kY, c * xPos + kX)] = color
#         return Object(points=newPoints)
#
#
#     def merge(self, object):
#         newPoints = {key:value for key,value in self.points.items()}
#         for yPos, xPos in object.points.keys():
#             assert (yPos, xPos) not in newPoints
#             newPoints[(yPos, xPos)] = object.points[(yPos, xPos)]
#         return Object(points=newPoints)
#
#     def split(self, isHorizontal):
#         if isHorizontal:
#             halfway = self.getNumRows() // 2
#             topHalf = Object(points={(y,x): self.points[(y,x)] for y,x in self.points.keys() if y < halfway})
#             if self.getNumRows() % 2 == 1:
#                 halfway += 1
#             bottomHalf = Object(points={(y-halfway,x): self.points[(y,x)] for y,x in self.points.keys() if y >= halfway})
#             return topHalf,bottomHalf
#         else:
#             halfway = self.getNumCols() // 2
#             leftHalf = Object(points={(y,x): self.points[(y,x)] for y,x in self.points.keys() if x < halfway})
#             if self.getNumCols() % 2 == 1:
#                 halfway += 1
#             rightHalf = Object(points={(y,x-halfway): self.points[(y,x)] for y,x in self.points.keys() if x >= halfway})
#             return leftHalf,rightHalf
#
#
#     def concat(self, object, direction):
#
#         if direction == 'right':
#             return self.merge(object.move(0, self.numCols))
#         elif direction == 'down':
#             return self.merge(object.move(self.numRows, 0))
#
#         elif direction == 'left':
#             return object.merge(self.move(0, object.numCols))
#         elif direction == 'up':
#             return object.merge(self.move(object.numRows, 0))
#
#         else:
#             print(direction)
#             raise NotImplementedError
#
    # def zipGrids(self, object, f):
    #     assert object.numCols == self.numCols and object.numRows == self.numRows
    #     assert set(object.points.keys()) == set(self.points.keys())
    #     return Object(points={key:f(self.points[key])(object.points[key]) for key in self.points.keys()})

#     def __len__(self):
#         return 100 * self.numRows + self.numCols
#
#     def __eq__(self, other):
#         if isinstance(other, Object):
#             sameKeys = set(other.points.keys()) == set(self.points.keys())
#             if sameKeys:
#                 allPointsEqual = all([self.points[key] == other.points[key] for key in self.points.keys()])
#                 return self.numRows == other.numRows and self.numCols == other.numCols and sameKeys and allPointsEqual
#         return False
#
#     def findBlocksByCorner(self, backgroundColor):
#
#         def dfs(v, edges, visited):
#
#             visited.add(v)
#             for adjacent in edges[v]:
#                 if adjacent not in visited:
#                     dfs(adjacent, edges, visited)
#             return
#
#         visited = set()
#         vertices = [v for v in self.points if self.points[v] != backgroundColor]
#         edges = self.createEdgeMap(backgroundColor)
#         blocks = []
#
#         for v in vertices:
#             if v not in visited:
#                 connected = set()
#                 dfs(v, edges, connected)
#                 blocks.append(connected)
#                 visited = visited.union(connected)
#
#         return [Object(points={key:self.points[key] for key in block}, backgroundColor=backgroundColor, isGrid=False) for block in blocks]
#
#     def createEdgeMap(self, backgroundColor):
#         edges = {}
#         for y,x in [key for key in self.points.keys()]:
#             if self.points[(y,x)] != backgroundColor:
#                 edges[(y, x)] = []
#                 for y_inc, x_inc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
#                     if (y + y_inc, x + x_inc) in self.points and self.points[(y + y_inc, x + x_inc)] != backgroundColor:
#                         edges[(y, x)] += [(y + y_inc, x + x_inc)]
#         return edges

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
# def _emptyGrid(a): return Object(mask=np.full((a.getNumRows(), a.getNumCols()), a.backgroundColor).tolist())


##### Blocks #####

def _head(blocks): return blocks[0]
def _mergeBlocks(blocks): return _reduce(lambda a: lambda b: a.merge(b))(Block(points={}, originalGrid=blocks[-1].originalGrid))(blocks)
def _getListBlock(blocks): return lambda i: blocks[i]
def _filterAndMinGrid(f): return lambda blocks: _blocksToMinGrid(_filter(f)(blocks))(False)

def _blocksToMinGrid(blocks): return lambda withOriginal: _mergeBlocks(blocks).toMinGrid(withOriginal=withOriginal)
def _blocksToGrid(blocks): return lambda withOriginal: lambda numRows: lambda numCols: _blockToGrid(_mergeBlocks(blocks))(withOriginal)(numRows)(numCols)
def _blocksAsGrid(blocks): return lambda withOriginal: _blockAsGrid(_mergeBlocks(blocks))(withOriginal)

def _sortBlocks(blocks): return lambda f: sorted(blocks, key=lambda block: f(block), reverse=True)
def _highestTileBlock(blocks): return _head(_sortBlocks(blocks)(_numTiles))

##### Block #####

def _fillIn(block): return lambda c: block.fillIn(c)
def _fill(block): return lambda c: block.fillPattern(c)
def _fillWithNthColor(block): return lambda n: block.fillPattern(_findNthColor(block)(n))
def _replaceColors(block): return lambda cOriginal: lambda cNew: block.replaceColors(cOriginal, cNew) 
def _replaceNthColors(block): return lambda nOriginal: lambda nNew: block.replaceColors(_findNthColor(block)(nOriginal), _findNthColor(block)(nNew))
def _reflect(a): return lambda isHorizontal: a.reflect(isHorizontal)
def _move(a): return lambda t: lambda direction: lambda keepOriginal: a.move(*{'down': (t, 0), 'up': (-t, 0), 'right': (0, t), 'left': (-t, 0)}[direction], keepOriginal=keepOriginal)
def _grow(a): return lambda n: a.grow(n)
def _concat(a): return lambda b: lambda dir: a.concat(b, dir)
def _concatN(a): return lambda b: lambda dir: lambda n: _concat(a)(b)(dir) if n <= 1 else _concatN(a.concat(b, dir))(b)(dir)(n-1)
def _concatUntilEdge(a): return lambda b: lambda dir: a.concatUntilEdge(b, dir)
def _duplicate(a): return lambda direction: a.concat(a.copy(), direction)
def _duplicateN(a): return lambda direction: lambda n: _concatN(a)(a)(direction)(n)
def _duplicateUntilEdge(a): return lambda direction: a.concatUntilEdge(a, direction)
def _concatNAndReflect(a): return lambda isHorizontal: lambda dir: _concatN(a)(_reflect(a)(isHorizontal))(dir)(1)
# def _fillAlternatingColsGrey(block): return block.fillPattern(_grey, mask=[block.points[key] for key in block.points.keys()])
def _duplicate2dN(a): return lambda n: _duplicateN(_duplicateN(a)('down')(n))('right')(n)
def _zipGrids(a): return lambda b: lambda f: a.zipGrids(b,f)
def _zipGrids2(grids): return lambda f: grids[0].zipGrids(grids[1], f)

def _isSymmetrical(block): return lambda horizontal: block.isSymmetrical(horizontal)
def _hasGeqNTiles(block): return lambda c: _numTiles(block) >= c
def _hasGeqNColors(block): return lambda n: _numColors(block) >= n

def _blockToMinGrid(block): return lambda withOriginal: block.toMinGrid(withOriginal=withOriginal)
def _blockToGrid(block): return lambda withOriginal: lambda numRows: lambda numCols: block.toGrid(numRows, numCols, withOriginal=withOriginal)
def _blockAsGrid(block): return lambda withOriginal: block.toGrid(block.originalGrid.getNumRows(), block.originalGrid.getNumCols(), withOriginal=withOriginal)
# def _blockAsSpecifiedGrid(block): return lambda grid: block.toGrid(grid.getNumRows(), grid.getNumCols())

def _split(a): return lambda isHorizontal: a.split(isHorizontal)

def _numTiles(block): return len([block.points[key] for key in block.points.keys()])
def _numColors(block): return len(set(block.points.values()))

##### Grid #####

def _findNonFullRectanglesBlackB(a): return a.findTouchingRectangles(0, full=False)
def _findRectanglesBlackB(a): return a.findTouchingRectangles(0)
def _findRectanglesByB(a): return lambda backgroundColor: a.findTouchingRectangles(backgroundColor)
def _findSameColorBlocks(a): return lambda boxBlocks: a.findSameColorBlocks(boxBlocks=boxBlocks)
def _findBlocksByColor(a): return lambda boxBlocks: lambda color: a.findBlocksBy(colors=[color], boxBlocks=boxBlocks)
def _findBlocksByCorner(a): return lambda boxBlocks: a.findBlocksBy(colors=[i for i in range(1,10)], isCorner=True, boxBlocks=boxBlocks)
def _findBlocksByEdge(a): return lambda boxBlocks: a.findBlocksBy(colors=[i for i in range(1,10)], isCorner=False, boxBlocks=boxBlocks)

def _findNthColor(a): return lambda n: np.argsort(np.bincount(list(a.points.values())))[-n]

def _splitAndMerge(a): return lambda f: lambda isHorizontal: _zipGrids2(_split(a)(isHorizontal))(f)

def _numRows(a): return a.getNumRows()
def _numCols(a): return a.getNumCols()

##### Color #####

def _keepNonBlacks(c): return lambda c2: c2 if c != _black else _black
def _keepBlackAnd(cNew): return lambda c: lambda c2: _black if (c2 == _black and c == _black) else cNew
def _keepBlackOr(cNew): return lambda c: lambda c2: _black if (c2 == _black or c == _black) else cNew
def _keepBlackXOr(cNew): return lambda c: lambda c2: _black if ((c2 == _black and not c == _black) or (c2 == _black and not c == _black)) else cNew


##### Any Type #####

def _filter(f): return lambda l: list(filter(f, l))
def _map(f): return lambda l: list(map(f, l))
def _reduce(f): return lambda x0: lambda l: reduce(lambda a, x: f(a)(x), l, x0)


##### Solutions #####

def _solvefcb5c309(grid): return _blockToMinGrid(_fill(_highestTileBlock(_findBlocksByColor(grid)(False)(_findNthColor(grid)(2))))(_findNthColor(grid)(3)))(True)
def _solve50cb2852(grid): return lambda c: _blocksAsGrid(_map(lambda block: _fillIn(block)(c))(_findRectanglesBlackB(grid)))(False)
def _solve0520fde7(a): return _zipGrids2(_split(a)(False))(_keepBlackOr(_red))
def _solve007bbfb7(a): return _zipGrids(_grow(a)(3))(_duplicate2dN(a)(2))(_keepNonBlacks)
def _solvec9e6f938(a): return _concatNAndReflect(a)(False)('right')
def _solve97999447(a): return _solveGenericBlockMap(a)(lambda grid: _findRectanglesBlackB(grid))(lambda block: (_duplicateUntilEdge(_concat(block)(_fill(block)(_grey))('right'))('right')))(lambda blocks: _blocksAsGrid(blocks)(False))
def _solvef25fbde4(a): return _solveGenericBlockMap(a)(lambda grid: _findBlocksByCorner(grid)(False))(lambda block: _grow(block)(2))(lambda block: _blocksToMinGrid(block)(False))
def _solve72ca375d(a): return _filterAndMinGrid(lambda block: _isSymmetrical(block)(False))(_findSameColorBlocks(a)(False))
def _solve5521c0d9(a): return _solveGenericBlockMap(a)(_findRectanglesBlackB)(lambda block: _move(block)(_numRows(block))('up')(False))(lambda blocks: _blocksAsGrid(blocks)(False))
def _solvece4f8723(a): return _splitAndMerge(a)(_keepBlackAnd(_green))(True)

#### Function Blueprints ####

def _solveGenericBlockMap(grid): return lambda findFunc: lambda mapFunc: lambda combineFunc: combineFunc(_map(mapFunc)(findFunc(grid)))
# def _solve42a50994(grid): return lambda count: _blocksToGrid(_filter(lambda block: _hasMinTiles(count)(block))(_findBlocksBy(grid)(True)))(grid.getNumRows())(grid.getNumCols())


class RecursionDepthExceeded(Exception):
    pass

if runFull:
    tdirection = baseType('direction')
    tdisplacement = baseType('tdisplacement')
    tcolor = baseType('color')
    tgrid = baseType('tgrid')
    tblock = baseType('tblock')
    tblocks = tlist(tblock)
    tgrids = tlist(tgrid)

def leafPrimitives():
    return [

        Primitive('1', tint, 1),
        Primitive('2', tint, 2),
        Primitive('3', tint, 3),
        Primitive('4', tint, 4),
        Primitive('5', tint, 5),
        Primitive('6', tint, 6),
        Primitive('7', tint, 7),
        Primitive('8', tint, 8),
        Primitive('9', tint, 9),

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
        Primitive("maroon", tcolor, _maroon)]


def basePrimitives():
    return [

##### tblocks #####

    # # arrow (tblocks, tblock)
    Primitive('head', arrow(tblocks, tblock), _head),
    Primitive('mergeBlocks', arrow(tblocks, tblock),  _mergeBlocks),
    Primitive('getListBlock', arrow(tblocks, tint, tblock), _getListBlock),
    # arrow(tblocks, tgrid)
    Primitive('blocksToMinGrid', arrow(tblocks, tbool, tgrid), _blocksToMinGrid),
    # Primitive('blocksToGrid',arrow(tblocks, tbool, tint, tint, tgrid), _blocksToGrid),
    Primitive("blocksAsGrid", arrow(tblocks, tbool, tgrid), _blocksAsGrid),
    Primitive("filterAndMinGrid", arrow(arrow(tblock, tbool), tblocks, tbool, tgrid), _filterAndMinGrid),
    # arrow(tblocks, tblocks)
    Primitive('sortBlocks',arrow(tblocks, arrow(tblock, tint), tblocks), _sortBlocks),
    Primitive("filterBlocks", arrow(arrow(tblock, tbool), tblocks, tblocks), _filter),
    Primitive("mapBlocks", arrow(arrow(tblock, tblock), tblocks, tblocks), _map),
    # arrow(tblocks, tint)
    Primitive('highestTileBlock', arrow(tblocks, tint), _highestTileBlock),

##### tblock ######

    # arrow(tblock, tblock)
    Primitive('fillIn', arrow(tblock, tcolor, tblock), _fillIn),
    Primitive('fill', arrow(tblock, tcolor, tblock), _fill),
    Primitive('fillWithNthColor', arrow(tblock, tint, tblock), _fillWithNthColor),
    Primitive('replaceColors', arrow(tblock, tcolor, tcolor, tblock), _replaceColors),
    Primitive('_replaceNthColors', arrow(tblock, tint, tint, tblock), _replaceNthColors),
    Primitive('reflect', arrow(tblock, tbool, tblock), _reflect),
    Primitive('move', arrow(tblock, tint, tdirection, tblock), _move),
    Primitive('grow', arrow(tblock, tint, tblock), _grow),
    Primitive('concat', arrow(tblock, tblock, tdirection, tblock), _concat),
    Primitive('concatN', arrow(tblock, tblock, tdirection, tint, tblock), _concatN),
    Primitive('concatUntilEdge', arrow(tblock, tblock, tdirection, tblock), _concatUntilEdge),
    Primitive('duplicate', arrow(tblock, tdirection, tblock), _duplicate),
    Primitive('duplicateN', arrow(tblock, tdirection, tint, tblock), _duplicateN),
    Primitive('duplicateUntilEdge', arrow(tblock, tdirection, tblock), _duplicateUntilEdge),
    Primitive('concatNAndReflect', arrow(tblock, tbool, tdirection, tblock), _concatNAndReflect),
    # arrow(tblock, tbool)
    Primitive('isSymmetrical', arrow(tblock, tbool, tbool), _isSymmetrical),
    Primitive('hasGeqNTiles', arrow(tblock, tint, tbool), _hasGeqNTiles),
    Primitive('hasGeqNcolors', arrow(tblock, tint, tbool), _hasGeqNColors), # (5117e062)
    # # arrow(tblock, tgrid)
    Primitive("blockToGrid", arrow(tblock, tint, tint, tbool, tgrid), _blockToGrid),
    Primitive("blockAsGrid", arrow(tblock, tbool, tgrid), _blockAsGrid),
    Primitive("blockToMinGrid", arrow(tblock, tbool, tgrid), _blockToMinGrid),
    # arrow(tblock, tblocks)
    Primitive('split', arrow(tblock, tbool, tblocks), _split),
    # arrow(tblock, tint, tcolor)
    Primitive('findNthBlockColor', arrow(tblock, tint, tcolor), _findNthColor),
    # arrow(tblock, tint)
    Primitive('numColors', arrow(tblock, tint), _numColors),
    Primitive('numTiles', arrow(tblock, tint), _numTiles),

##### tcolor ######

    # arrow(tcolor, tcolor)
    Primitive('keepNonBlacks', arrow(tcolor, tcolor, tcolor), _keepNonBlacks),
    Primitive('keepBlackOr', arrow(tcolor, tcolor, tcolor, tcolor), _keepBlackOr),
    Primitive('keepBlackAnd', arrow(tcolor, tcolor, tcolor, tcolor), _keepBlackAnd),

##### tgrid #####

    # arrow(tgrid, tblocks)
    # Primitive('findRectanglesBlackB', arrow(tgrid, tblocks), _findRectanglesBlackB),
    # Primitive('findRectanglesByB', arrow(tgrid, tcolor, tblocks), _findRectanglesByB),
    # Primitive('findBlocksByColor', arrow(tgrid, tcolor, tbool, tblocks), _findBlocksByColor),
    # Primitive('findBlocksByCorner', arrow(tgrid, tbool, tblocks), _findBlocksByCorner),
    # Primitive('findBlocksByEdge', arrow(tgrid, tbool, tblocks), _findBlocksByEdge),

    # #arrow(tgrid, tblock)
    Primitive('gridToBlock', arrow(tgrid, tblock), lambda grid: grid),
    # arrow(tgrid, grid)
    # Primitive('reflect', arrow(tgrid, tbool, tgrid), _reflect),
    # Primitive('grow', arrow(tgrid, tint, tgrid), _grow),
    # Primitive('concat', arrow(tgrid, tgrid, tdirection, tgrid), _concat),
    # Primitive('concatN', arrow(tgrid, tgrid, tdirection, tint, tgrid), _concatN),
    # Primitive('duplicate', arrow(tgrid, tdirection, tgrid), _duplicate),
    # Primitive('duplicateN', arrow(tgrid, tdirection, tint, tgrid), _duplicateN),
    # Primitive('zipGrids', arrow(tgrid, tgrid, arrow(tcolor, tcolor, tcolor), tgrid), _zipGrids),
    # Primitive('zipGrids2', arrow(tgrids, arrow(tcolor, tcolor, tcolor), tgrid), _zipGrids2),
    # Primitive('concatNAndReflect', arrow(tgrid, tbool, tdirection, tgrid), _concatNAndReflect),
    # Primitive('solve0520fde7', arrow(tgrid, tgrid), _solve0520fde7),
    # Primitive('solve007bbfb7', arrow(tgrid, tgrid), _solve007bbfb7),
    # Primitive('solve50cb2852', arrow(tgrid, tcolor, tgrid), _solve50cb2852),
    # Primitive('solvefcb5c309', arrow(tgrid, tgrid), _solvefcb5c309),
    # Primitive('solvec9e6f938', arrow(tgrid, tgrid), _solvec9e6f938),
    # Primitive('solve97999447', arrow(tgrid, tgrid), _solve97999447),
    # Primitive('solvef25fbde4', arrow(tgrid, tgrid), _solvef25fbde4),
    
    # Primitive('solve72ca375d', arrow(tgrid, tgrid), _solve72ca375d),
    # Primitive('solve5521c0d9', arrow(tgrid, tgrid), _solve5521c0d9),
    # Primitive('solvece4f8723', arrow(tgrid, tgrid), _solvece4f8723),
    # arrow(tgrid, tcolor)
    Primitive('findNthGridColor', arrow(tgrid, tint, tcolor), _findNthColor),
    # arrow(tgrid, tgrid)
    # Primitive('splitAndMergeGrid', arrow(tgrid, arrow(tcolor, tcolor, tcolor), tbool, tgrid), _splitAndMerge),
    # arrow(tgrid, tint)
    Primitive('numRows', arrow(tgrid, tint), _numRows),
    Primitive('numCols', arrow(tgrid, tint), _numCols),

#####

##### tgrid #####

    # t0
    # Primitive("reduce", arrow(arrow(t1, t0, t1), t1, tlist(t0), t1), _reduce),
    # Primitive("map", arrow(arrow(t0, t1), tlist(t0), tlist(t1)), _map),
    # Primitive("filter", arrow(arrow(t0, tbool), tlist(t0), tlist(t0)), _filter),
    # Primitive("mapi",arrow(arrow(tint,t0,t1), tlist(t0), tlist(t1)),_mapi),
    # Primitive("filteri", arrow(arrow(tint, t0, t1), tlist(t0), tlist(t1)), _filteri)]

##### Task Blueprints #####

    # Primitive('findAndMap', arrow(tgrid, arrow(tgrid, tblocks), arrow(tblock, tblock), arrow(tblocks, tgrid), tgrid), _solveGenericBlockMap)
    ]



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

def getTask(filename, directory):
    with open(directory + '/' + filename, "r") as f:
        loaded = json.load(f)

    train = [((Grid(gridArray=example['input'],)), Grid(gridArray=example['output'])) for example in loaded['train']]
    test = [((Grid(gridArray=example['input'],)), Grid(gridArray=example['output'])) for example in loaded['test']]

    return train, test

if __name__ == "__main__":

    directory = '/'.join(os.path.abspath(__file__).split('/')[:-4]) + '/arc-data/data/training'
    train,test = getTask('a416b8f3.json', directory)

    for i in range(len(train)):
        print('\nExample {}'.format(i))
        grid, outputGrid = train[i]
        got = _blockToMinGrid(_move(grid)(_numCols(grid))('right')(True))(False)
        # for block in got:
            # block.pprint()
        #     isHorizontal = False
        #     reflectedBlock = block.reflect(isHorizontal)
        #     refA, refB = reflectedBlock.split(isHorizontal)
        #     refA.pprint()
        #     a,b = block.split(isHorizontal)
        #     a.pprint()
        #
        #     refB.pprint()
        #     b.pprint()
        #     print(block.isSymmetrical(isHorizontal))
        #     print('\n')

        # print('Input: ')
        # inputGrid.pprint()
        print('Got: ')
        got.pprint()
        print('Expected')
        outputGrid.pprint()
        if (got == outputGrid):
            print('HIT')
        # break
        # inputGrid.pprint()
        # count = 0

        # res = _filter(lambda block: _hasMinTiles(4)(block))(inputGrid.findBlocksByCorner(0))
        # _blocksToGrid(res).pprint()
