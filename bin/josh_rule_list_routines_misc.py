import matplotlib
matplotlib.use('Agg')

try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

import glob
import json
import random
import itertools
import math
import numpy as np
import pandas as pd
import re
import subprocess
from ast import literal_eval
from dreamcoder.program import Program, Primitive, prettyProgram
from dreamcoder.type import *
from dreamcoder.grammar import Grammar
from functools import reduce
from joblib import Parallel, delayed, parallel_backend

# set the seed first thing
# random.seed(1)

# Define ourselves a failure mode for recursion
class RecursionDepthExceeded(Exception):
    pass

# notice that these are curried
def _reverse(x): return list(reversed(x))
def _cons(x): return lambda xs: [x] + xs
def _append(xs): return lambda x: xs + [x]
def _single(x): return [x]
def _concat(x): return lambda y: x + y
def _unique(x): return list(dict.fromkeys(x))
def _product(x): return reduce(lambda x,y: x*y, x, 1)
def _first(x): return x[0]
def _second(x): return x[1]
def _third(x): return x[2]
def _nth(i):
    if i > 0:
        return lambda x: x[i-1]
    else:
        raise IndexError
def _repeat(x): return lambda n: [x]*n
def _range(start): return lambda step: lambda stop: list(range(start, stop+1 if step > 0 else stop-1, step))
def _last(x): return x[-1]
def _drop(i): return lambda xs: xs[i:]
def _droplast(i): return lambda xs: xs[:-i] if i > 0 else xs[:]
def _take(i): return lambda xs: xs[:i]
def _takelast(i): return lambda xs: xs[-i:] if i > 0 else []
def _eq(x): return lambda y: x == y
def _is_empty(x): 
    return len(x) == 0
def _mod(x): return lambda y: x % y
def _slice(x): return lambda y: lambda l: l[(x-1):y]
def _cut_idx(i): return lambda xs: xs[:(i-1)] + xs[i:]
def _cut_slice(i):
    def helper(j):
        if i > j:
            raise IndexError
        return lambda xs: xs[:(i-1)] + xs[j:]
    return helper
def _cut_val(v):
    def helper(xs):
        result = []
        found = False
        for x in xs:
            if x != v or found:
                result.append(x)
            elif x == v:
                found = True
        return result
    return helper
def _cut_vals(v): return lambda xs: [x for x in xs if x != v]
def _replace(idx): return lambda y: lambda xs: [y if i == idx else x for i, x in enumerate(xs, 1)]
def _flatten(l): return [x for xs in l for x in xs]
def _map(f): return lambda l: list(map(f, l))
def _if(c): return lambda t: lambda f: t if c else f
def _addition(x): return lambda y: x + y
def _subtraction(x): return lambda y: x - y
def _multiplication(x): return lambda y: x * y
def _division(x):
    def helper(y):
        if y == 0:
            raise ValueError
        return x // y
    return helper
def _gt(x): return lambda y: x > y
def _lt(x): return lambda y: x < y
# not the most general form (i.e. zip-with) but it matches standard usage
def _zip(xs): return lambda ys: [list(x) for x in zip(xs, ys)]
def _mapi(f): return lambda l: list(map(lambda i_x: f(i_x[0])(i_x[1]), enumerate(l, 1)))
def _and(x): return lambda y: x and y
def _or(x): return lambda y: x or y
def _not(x): return not x
def _group(key):
    def helper(xs):
        keys = []
        groups = {}
        for x in xs:
            k = key(x)
            if k not in groups:
                keys.append(k)
                groups[k] = [x]
            else:
                groups[k].append(x)
        return [groups[k] for k in keys]
    return helper
def _is_even(x): return x % 2 == 0
def _is_odd(x): return x % 2 == 1
def _count(p): return lambda xs: sum(p(x) for x in xs)
def _filter(f): return lambda xs: list(filter(f, xs))
def _filteri(f): return lambda xs: [x for i, x in enumerate(xs, 1) if f(i)(x)]
def _fold(f): return lambda x0: lambda xs: reduce(lambda a, x: f(a)(x), xs, x0)
def _foldi(f): return lambda x0: lambda xs: reduce(lambda a, t: f(t[0])(a)(t[1]), enumerate(xs, 1), x0)
def _is_in(xs): return lambda x: x in xs
def _find(p): return lambda xs: [i for i, x in enumerate(xs, 1) if p(x)]
def _insert(x): return lambda i: lambda xs: xs[:(i-1)] + [x] + xs[(i-1):]
def _splice(x): return lambda i: lambda xs: xs[:(i-1)] +  x  + xs[(i-1):]
def _tail(xs): return xs[1:]
def _swap(i):
    def swap_helper_j(j):
        def swap_helper_xs(xs):
            fst = min(i,j)
            snd = max(i,j)
            return xs[:(fst-1)] + [xs[(snd-1)]] + xs[fst:(snd-1)] + [xs[(fst-1)]] + xs[snd:]
        return swap_helper_xs
    return swap_helper_j
def _sort(k): return lambda xs: sorted(xs, key=k)
def _fix(argument):
    def inner(body):
        recursion_limit = [50]

        def fix(x):
            def r(z):
                recursion_limit[0] -= 1
                if recursion_limit[0] <= 0:
                    raise RecursionDepthExceeded()
                else:
                    return fix(z)

            return body(r)(x)
        return fix(argument)

    return inner

def model_comparison_primitives_9():
    return _model_comparison_primitives(9)

def model_comparison_primitives_99():
    return _model_comparison_primitives(99)

def _model_comparison_primitives(max_num):
    return [Primitive(str(j), tint, j) for j in range(0,max_num+1)] + [
        Primitive("nan", tint, math.nan),
        Primitive("true", tbool, True),
        Primitive("false", tbool, False),
        Primitive("empty", tlist(t0), []),
        Primitive("cons", arrow(t0, tlist(t0), tlist(t0)), _cons),
        Primitive("+", arrow(tint, tint, tint), _addition),
        Primitive("-", arrow(tint, tint, tint), _subtraction),
        Primitive(">", arrow(tint, tint, tbool), _gt),
        Primitive("fix", arrow(t0, arrow(arrow(t0, t1), t0, t1), t1), _fix),
        Primitive("head", arrow(tlist(t0), t0), _first),
        Primitive("if", arrow(tbool, t0, t0, t0), _if),
        Primitive("is_empty", arrow(t0, t0, tbool), _is_empty),
        Primitive("is_equal", arrow(t0, t0, tbool), _eq),
        # `lambda` is built into the representation.
        Primitive("tail", arrow(tlist(t0), tlist(t0)), _tail),
    ]

# define some primitives
def primitives():
    return [Primitive(str(j), tint, j) for j in range(-2,100)] + [
        Primitive("%", arrow(tint, tint, tint), _mod),
        Primitive("*", arrow(tint, tint, tint), _multiplication),
        Primitive("+", arrow(tint, tint, tint), _addition),
        Primitive("-", arrow(tint, tint, tint), _subtraction),
        Primitive("/", arrow(tint, tint, tint), _division),
        Primitive("<", arrow(tint, tint, tbool), _lt),
        Primitive("==", arrow(t0, t0, tbool), _eq),
        Primitive(">", arrow(tint, tint, tbool), _gt),
        Primitive("abs", arrow(tint, tint), abs),
        Primitive("and", arrow(tbool, tbool, tbool), _and),
        Primitive("append", arrow(tlist(t0), t0, tlist(t0)), _append),
        Primitive("concat", arrow(tlist(t0), tlist(t0), tlist(t0)), _concat),
        Primitive("cons", arrow(t0, tlist(t0), tlist(t0)), _cons),
        Primitive("count", arrow(arrow(t0, tbool), tlist(t0), tint), _count),
        Primitive("cut_idx", arrow(tint, tlist(t0), tlist(t0)), _cut_idx),
        Primitive("cut_slice", arrow(tint, tint, tlist(t0), tlist(t0)), _cut_slice),
        Primitive("cut_val", arrow(t0, tlist(t0), tlist(t0)), _cut_val),
        Primitive("cut_vals", arrow(t0, tlist(t0), tlist(t0)), _cut_vals),
        Primitive("drop", arrow(tint, tlist(t0), tlist(t0)), _drop),
        Primitive("droplast", arrow(tint, tlist(t0), tlist(t0)), _droplast),
        Primitive("empty", tlist(t0), []),
        Primitive("false", tbool, False),
        Primitive("filter", arrow(arrow(t0, tbool), tlist(t0), tlist(t0)), _filter),
        Primitive("filteri", arrow(arrow(tint, t0, tbool), tlist(t0), tlist(t0)), _filteri),
        Primitive("flatten", arrow(tlist(tlist(t0)), tlist(t0)), _flatten),
        Primitive("fold", arrow(arrow(t1, t0, t1), t1, tlist(t0), t1), _fold),
        Primitive("foldi", arrow(arrow(tint, t1, t0, t1), t1, tlist(t0), t1), _foldi),
        Primitive("group", arrow(arrow(t0, t1), tlist(t1), tlist(tlist(t1))), _group),
        Primitive("first", arrow(tlist(t0), t0), _first),
        Primitive("second", arrow(tlist(t0), t0), _second),
        Primitive("third", arrow(tlist(t0), t0), _third),
        Primitive("if", arrow(tbool, t0, t0, t0), _if),
        Primitive("is_even", arrow(tint, tbool), _is_even),
        Primitive("is_odd", arrow(tint, tbool), _is_odd),
        Primitive("last", arrow(tlist(t0), t0), _last),
        Primitive("length", arrow(tlist(t0), tint), len),
        Primitive("map", arrow(arrow(t0, t1), tlist(t0), tlist(t1)), _map),
        Primitive("mapi", arrow(arrow(tint, t0, t1), tlist(t0), tlist(t1)), _mapi),
        Primitive("max", arrow(tlist(t0), tint), max),
        Primitive("min", arrow(tlist(t0), tint), min),
        Primitive("not", arrow(tbool, tbool), _not),
        Primitive("nth", arrow(tint, tlist(t0), t0), _nth),
        Primitive("or", arrow(tbool, tbool, tbool), _or),
        Primitive("product", arrow(tlist(tint), tint), _product),
        Primitive("range", arrow(tint, tint, tint, tlist(tint)), _range),
        Primitive("repeat", arrow(t0, tint, tlist(t0)), _repeat),
        Primitive("replace", arrow(tint, t0, tlist(t0), tlist(t0)), _replace),
        Primitive("reverse", arrow(tlist(t0), tlist(t0)), _reverse),
        Primitive("singleton", arrow(t0, tlist(t0)), _single),
        Primitive("slice", arrow(tint, tint, tlist(t0), tlist(t0)), _slice),
        Primitive("sort", arrow(arrow(t0, tint), tlist(t0), tlist(t0)), _sort),
        Primitive("sum", arrow(tlist(tint), tint), sum),
        Primitive("take", arrow(tint, tlist(t0), tlist(t0)), _take),
        Primitive("takelast", arrow(tint, tlist(t0), tlist(t0)), _takelast),
        Primitive("true", tbool, True),
        Primitive("unique", arrow(tlist(t0), tlist(t0)), _unique),
        Primitive("zip", arrow(tlist(t0), tlist(t0), tlist(tlist(t0))), _zip),
        Primitive("is_in", arrow(tlist(t0), t0, tbool), _is_in),
        Primitive("find", arrow(arrow(t0, tbool), tlist(t0), tlist(tint)), _find),
        Primitive("insert", arrow(t0, tint, tlist(t0), tlist(t0)), _insert),
        Primitive("splice", arrow(tlist(t0), tint, tlist(t0), tlist(t0)), _splice),
        Primitive("swap", arrow(tint, tint, tlist(t0), tlist(t0)), _swap),
    ]

def proportion(xs, f):
    return sum(f(i, o) for i,o in xs)/len(xs)

def proportion_set(xs, f):
    return len({f(i, o) for i,o in xs})/len(xs)

def limit(xs, accept, f):
    return max(0, sum(f(i, o) for i,o in xs) - accept)

def forbid(xs, f):
    return limit(xs, 0, f)

def center(xs, f, factor = 1/2):
    return 1 + abs(factor * len(xs) - sum(f(i,o) for i, o in xs))

def proportion_unique_elements(xs):
   return sum(len(set(i)) for i,o in xs) / sum(len(i) for i,o in xs)

def wave_pilot():
    return [
        {"concept": '(lambda (unique $0))',
         "adjust": lambda xs: min(1.0, 1.0/(len(xs)-2)*sum((len(o)/len(i) < 0.75 if len(i) > 0 else 1) for i, o in xs)),
         "inputs": [
                 [7, 31, 7, 7, 31],
                 [3, 8, 3],
                 [7, 9, 2, 2, 3, 7, 6, 7],
                 [19, 19],
                 [66, 3, 89, 4, 66, 66, 4, 37, 0, 3],
                 [56, 93, 1, 1, 0, 93],
                 [],
                 [19, 38, 14, 76, 7, 4, 88],
                 [16, 25, 8, 8],
                 [79],
                 [5, 19, 49, 7, 62]
         ]},
        {"concept": '(lambda (singleton (length $0)))',
         "adjust": lambda xs: 1.0,
         "inputs": [
             [],
             [31],
             [23, 6],
             [38, 4, 18],
             [88, 67, 0, 44],
             [3, 3, 7, 49, 6],
             [80, 70, 51, 5, 98, 2],
             [45, 76, 37, 3, 8, 1, 76],
             [66, 12, 43, 12, 25, 6, 6, 15],
             [22, 24, 58, 84, 3, 46, 0, 22, 3],
             [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
         ]},
        {"concept": '(lambda (repeat (max $0) (min $0)))',
         "adjust": lambda xs: 1.0 if any(len(o) == 0 for i, o in xs) else 0.0,
         "inputs": [
                 [99, 7, 55], # 7/3
                 [36, 22, 2, 15, 7], # 2/5
                 [62, 5], # 5/2
                 [23, 9, 14, 7, 2, 31, 4, 4, 0, 18], # 0/10
                 [3, 3, 3, 3], # 3/4
                 [4, 4, 4], # 4/3
                 [32, 14, 67, 32, 9, 70, 77], # 9/7
                 [7], # 7/1
                 [12, 42, 92, 58, 62, 38], # 12/6
                 [48, 56, 39, 58, 13], # 13/5
                 [43, 84, 8, 17, 8, 78, 64, 10], # 8/9
         ]},
        {"concept": '(lambda (concat (reverse (drop 1 $0)) $0))',
         "adjust": lambda xs: 1.0,
         "inputs": [
             [],
             [1],
             [7, 7],
             [49, 0, 34],
             [54, 6, 3, 8],
             [70, 70, 3, 70, 3],
             [64, 15, 92, 54, 15, 85],
             [61, 6, 6, 2, 2, 6, 6],
             [0, 1, 1, 21, 4, 50, 50, 78],
             [93, 93, 93, 93, 93, 93, 93, 93, 93],
             [1, 79, 0, 21, 4, 32, 42, 81, 23, 9],
         ]},
        {"concept": '(lambda (concat (drop (last $0) $0) (take (last $0) $0)))',
         "adjust": lambda xs: 0 if sum(i[-1] >= len(i) for i, o in xs) > 2 else 1,
         "inputs": [
                 [1, 17, 4, 2],
                 [20, 14, 66, 2, 68, 46, 93, 5],
                 [50, 71, 6, 32, 1],
                 [72, 8, 54, 98, 72, 43, 49, 42, 7, 8],
                 [12, 5, 83, 5, 0, 1],
                 [46, 69, 70, 4, 20, 5, 42, 41, 22, 6],
                 [9, 33, 0],
                 [0, 23, 17, 81, 87, 3],
                 [53, 22, 57, 37, 59, 66, 26, 21, 4],
                 [96, 32, 99, 98, 98, 60, 80, 90, 26, 7],
                 [88, 10, 1, 78, 56, 32],
         ]},
        {"concept": '(lambda (flatten (map (lambda (cons (first $0) (singleton (length $0)))) (group (lambda $0) $0))))',
         "adjust": lambda xs: len({e for i, o in xs for e in o[1::2]})/10,
         "inputs": [
                 [2, 2, 2, 19, 2, 2, 25, 2],
                 [4, 4, 8, 4, 3],
                 [4, 4, 4, 4, 4, 4, 4],
                 [79, 79, 8, 79, 7, 7, 7, 79, 8],
                 [86, 86, 1, 1, 86, 1],
                 [8, 9, 98, 4, 7, 86],
                 [1, 41, 6, 90],
                 [33, 24, 0, 0, 1, 7, 33, 10],
                 [97, 18, 67, 67],
                 [8, 8, 9, 8, 1, 9, 8],
                 [0, 45, 7, 37, 94, 94, 7, 7, 45, 45],
         ]},
        {"concept": '(lambda (fold (lambda (lambda (if (> $0 (last $1)) (append $1 $0) $1))) (take 1 $0) (drop 1 $0)))',
         "adjust": lambda xs: 2*len({len(o) for i, o in xs})/11,
         "inputs": [
                 [1, 3, 2, 5, 3, 4, 7, 6, 9], #9
                 [22, 6, 7, 38, 62, 44, 78, 91], #8
                 [0, 4, 9], # 3
                 [5, 2, 19, 18, 37], #5
                 [4, 0, 9], # 3
                 [11, 23, 34, 55, 87], # 5
                 [97, 13, 82, 4, 55, 97, 3], #7
                 [], # 0
                 [34, 35, 62, 24, 75, 6], #6
                 [2, 6, 2, 10, 17, 3, 53, 9, 72, 3], # 10
                 [48, 61, 37, 86], #4
         ]},
        {"concept": '(lambda (fold (lambda (lambda (if (is_even (second $0)) (append $1 (first $0)) $1))) empty (zip (droplast 1 $0) (drop 1 $0))))',
         "adjust": lambda xs: len({len(o) for i, o in xs})/10,
         "inputs": [
                 [6, 0, 7, 32],
                 [62, 8, 59, 88, 98, 6],
                 [1, 96, 1, 13, 86, 77, 6, 10, 7, 0],
                 [6],
                 [1, 7],
                 [43, 4, 64, 5, 0],
                 [0, 2, 3],
                 [7, 14, 7, 6, 8, 57, 10],
                 [27, 6, 21, 6, 86, 8, 0],
                 [4, 10, 6, 8],
                 [6, 0, 85, 7, 10, 69, 22, 5],
         ]},
    ]

def human_experiments_wave_1():
    return [
        {
            "concept": "(lambda (cons 11 (cons 19 (cons 24 (cons 33 (cons 42 (cons 5 (cons 82 (cons 0 (cons 64 (cons 9 empty)))))))))))",
            "adjust": lambda xs: 0.0,
        },
        {
            "concept": "(lambda $0)",
            "adjust": lambda xs: - limit(xs, 1, lambda i,o: len(i) >= 7) + proportion_unique_elements(xs),
        },
        {
            "concept": "(lambda (singleton (length $0)))",
            "adjust": lambda xs: proportion_set(xs, lambda i,o: o[0]) + proportion_unique_elements(xs),
        },
        {
            "concept": "(lambda (singleton (max $0)))",
            "adjust": lambda xs: 2 * proportion_set(xs, lambda i,o: o[0]) + proportion_unique_elements(xs)
        },
        {
            "concept": "(lambda (splice (drop 1 (droplast 1 $0)) 2 $0))",
            "adjust": lambda xs: proportion(xs, lambda i,o: 6 >= len(i) >= 3) + proportion_unique_elements(xs),
        },
        {
            "concept": "(lambda (sort (lambda $0) $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) - limit(xs, 3, lambda i,o: len(i) <= 3 or len(i) >= 7),
        },
        {
            "concept": "(lambda (unique $0))",
            "adjust": lambda xs: proportion(xs, lambda i,o: (len(i) - len(o)) > 2) + proportion_set(xs, lambda i,o: len(i) - len(o)) + proportion_set(xs, lambda i,o: len(o)) - limit(xs, 2, lambda i,o: len(o) >= 7),
        },
        {
            "concept": "(lambda (singleton (sum $0)))",
            "adjust": lambda xs: 4 * proportion_set(xs, lambda i,o: o[0]) + 2 * proportion_set(xs, lambda i,o: len(i)) + proportion_set(xs, lambda i,o: o[0]/10) + proportion_unique_elements(xs),
        },
        {
            "concept": "(lambda (singleton (product $0)))",
            "adjust": lambda xs: 4 * proportion_set(xs, lambda i,o: o[0]) + 2 * proportion_set(xs, lambda i,o: len(i)) + proportion_set(xs, lambda i,o: o[0]/5) + proportion_unique_elements(xs),
        },
        {
            "concept": "(lambda (takelast 3 (sort (lambda $0) $0)))",
            "adjust": lambda xs: 2 * len({oe for i,o in xs for oe in o})/sum(len(o) for i,o in xs) + proportion_unique_elements(xs)
        },
        {
            "concept": "(lambda (repeat (max $0) (min $0)))",
            "adjust": lambda xs: proportion(xs, lambda i,o: min(i) <= 10) + proportion_set(xs, lambda i,o: max(i)) + proportion_set(xs, lambda i,o: min(i)) + proportion_unique_elements(xs)
        },
        {
            "concept": "(lambda (range 1 1 (last $0)))",
            "adjust": lambda xs: proportion(xs, lambda i,o: len(i) > 0 and i[-1] <= 10) + proportion_set(xs, lambda i,o: i[-1] if len(i) > 0 else 0) + proportion_unique_elements(xs)
        },
        {
            "concept": "(lambda (filter (lambda (> (first $1) (% $0 10))) $0))",
            "adjust": lambda xs: - limit(xs, 2, lambda i,o: len(o) <= 2 or len(o) >= 7) + proportion(xs, lambda i,o: len(i) > 1 and i[0] < 10) + proportion_set(xs, lambda i,o: i[0] if len(i) > 0 else 0) + proportion_unique_elements(xs)
        },
        {
            "concept": "(lambda (cons (last $0) $0))",
            "adjust": lambda xs: - limit(xs, 2, lambda i,o: len(o) <= 2 or len(o) >= 7) + proportion_unique_elements(xs) + proportion_set(xs, lambda i,o: i[-1])
        },
        {
            "concept": "(lambda (cons (sum (unique $0)) (append (unique $0) (sum (unique $0)))))",
            "adjust": lambda xs: proportion_set(xs, lambda i,o: len(o)) + proportion_set(xs, lambda i,o: o[0]) - limit(xs, 2, lambda i,o: len(o) <= 3 or len(o) >= 8) + 2 * proportion(xs, lambda i,o: (len(i) - len(o)) > 2) + 2 * proportion_set(xs, lambda i,o: len(i) - len(o))
        },
        {
            "concept": "(lambda (concat (reverse (drop 1 $0)) $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) - limit(xs, 2, lambda i,o: len(o) >= 10)
        },
        {
            "concept": "(lambda (concat (drop 3 $0) (take 3 $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) >= 3) - limit(xs, 2, lambda i,o: len(o) > 7)
        },
        {
            "concept": "(lambda (concat (drop (last $0) $0) (take (last $0) $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs) - limit(xs, 2, lambda i,o: len(o) > 7) + 4 * proportion(xs, lambda i,o: len(i) > 0 and len(i) > i[-1]) + proportion_set(xs, lambda i,o: i[-1] if len(i) else 0) + proportion_set(xs, lambda i,o: len(i) - i[-1])
        },
        {
            "concept": "(lambda ((lambda (concat ($0 first) (concat $1 ($0 last)))) (lambda (if (== ($0 $1) 8) empty (singleton 8)))))",
            "adjust": lambda xs: 2 / center(xs, lambda i,o: len(i) > 0 and i[0] == 8) + 2 / center(xs, lambda i,o: len(i) > 0 and i[-1] == 8) - limit(xs, 2, lambda i,o: len(o) >= 7) + proportion_unique_elements(xs)
        },
        {
            "concept": "(lambda (singleton (first $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion_set(xs, lambda i,o: o[0])
        },
        {
            "concept": "(lambda (singleton (last $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion_set(xs, lambda i,o: o[0])
        },
        {
            "concept": "(lambda (singleton (second (reverse $0))))",
            "adjust": lambda xs: 2 * proportion(xs, lambda i,o: len(i) >= 2) + proportion_unique_elements(xs) + proportion_set(xs, lambda i,o: o[0])
        },
        {
            "concept": "(lambda (singleton (nth (last $0) $0)))",
            "adjust": lambda xs: 2 * proportion(xs, lambda i,o: len(i) > 0) + proportion_unique_elements(xs) + proportion_set(xs, lambda i,o: o[0]) - 2 * limit(xs, 1, lambda i,o: len(i) > 0 and len(i) == i[-1])
        },
        {
            "concept": "(lambda (singleton (nth (nth (first $0) $0) $0)))",
            "adjust": lambda xs: 2 * proportion(xs, lambda i,o: len(i) > 0 and i[0] < len(i) and i[i[0]] < len(i)) + proportion_unique_elements(xs) + proportion_set(xs, lambda i,o: o[0])
        },
        {
            "concept": "(lambda (filter (lambda (== (/ (first $1) 10) (/ $0 10))) $0))",
            "adjust": lambda xs: proportion_set(xs, lambda i,o: i[0]/10 if len(i) else 0) + proportion_set(xs, lambda i,o: len(i) - len(o)) - limit(xs, 1, lambda i,o: len(o) <= 1 or len(o) == len(i)) + proportion_set(xs, lambda i,o: len(set(o))) + proportion_unique_elements(xs)
        },
        {
            "concept": "(lambda (drop 1 $0))",
            "adjust": lambda xs: proportion_unique_elements(xs)
        },
        {
            "concept": "(lambda (droplast 1 $0))",
            "adjust": lambda xs: proportion_unique_elements(xs)
        },
        {
            "concept": "(lambda (sort (lambda $0) (cut_idx 3 (drop 2 $0))))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 5)
        },
        {
            "concept": "(lambda (slice (first $0) (second $0) (drop 2 $0)))",
            "adjust": lambda xs: 4 * proportion(xs, lambda i,o: len(i) > 2 and 1 <= i[0] <= i[1] <= len(i)-2) + proportion_unique_elements(xs) - limit(xs, 1, lambda i,o: len(i) > 0 and i[0] == 1) - limit(xs, 0, lambda i,o: len(i) > 0 and i[0] == 0) - limit(xs, 1, lambda i,o: len(i) > 1 and i[1] == i[0]) - limit(xs, 1, lambda i,o: len(i) > 1 and i[1] == len(i)-2) - limit(xs, 0, lambda i,o: len(i) > 1 and i[1] < i[0]) - limit(xs, 0, lambda i,o: len(i) > 1 and i[1] > len(i)-2)
        },
        {
            "concept": "(lambda (take (first $0) (drop 1 $0)))",
            "adjust": lambda xs: 4 * proportion(xs, lambda i,o: len(i) > 1 and i[0] <= len(i)-1) + proportion_unique_elements(xs) + proportion_set(xs, lambda i,o: len(o)) + proportion_set(xs, lambda i,o: len(i)-len(o)) - limit(xs, 1, lambda i,o: len(i) > 0 and i[0] == len(i) - 1) - limit(xs, 1, lambda i,o: len(i) > 0 and i[0] == 0)
        },
        {
            "concept": "(lambda (filter (lambda (is_even (/ $0 10))) $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len({ie//10 for ie in i})/max(1,len(i)) > 4) + proportion_set(xs, lambda i,o: len(i)-len(o))
        },
        {
            "concept": "(lambda (cut_idx 3 $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) + 2 * proportion(xs, lambda i,o: len(i) > 3) - limit(xs, 2, lambda i,o: len(o) >= 7)
        },
        {
            "concept": "(lambda (cut_slice 2 5 $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) + 2 * proportion(xs, lambda i,o: len(i) > 5) - limit(xs, 2, lambda i,o: len(o) >= 7)
        },
        {
            "concept": "(lambda (cut_slice (first $0) (second $0) $0))",
            "adjust": lambda xs: 4 * proportion(xs, lambda i,o: len(i) > 2 and 1 <= i[0] <= i[1] <= len(i)) + proportion_unique_elements(xs) - limit(xs, 3, lambda i,o: len(i) > 0 and i[0] == 1) - limit(xs, 0, lambda i,o: len(i) > 0 and i[0] == 0) - limit(xs, 1, lambda i,o: len(i) > 1 and i[1] == i[0]) - limit(xs, 2, lambda i,o: len(i) > 1 and i[1] == len(i)) - limit(xs, 0, lambda i,o: len(i) > 1 and i[1] < i[0]) - limit(xs, 0, lambda i,o: len(i) > 1 and i[1] > len(i))
        },
        {
            "concept": "(lambda (cut_val 7 $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) + 2 * proportion(xs, lambda i,o: 7 in i) + 1 / center(xs, lambda i,o: i.count(7) > 1, factor = 8/11) + 1 / center(xs, lambda i,o: i.count(7) > 2, factor = 4/11)
        },
        {
            "concept": "(lambda (cut_val (max $0) $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) + 2 * proportion(xs, lambda i,o: len(i) > 1) + 1 / center(xs, lambda i,o: len(i) > 0 and i.count(max(i)) > 1, factor = 8/11) + 1 / center(xs, lambda i,o: len(i) > 0 and i.count(max(i)) > 2, factor = 4/11)
        },
        {
            "concept": "(lambda (cut_vals 3 $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) + 2 * proportion(xs, lambda i,o: 3 in i) + 1 / center(xs, lambda i,o: i.count(3) > 1, factor = 8/11) + 1 / center(xs, lambda i,o: i.count(3) > 2, factor = 4/11)
        },
        {
            "concept": "(lambda (cut_vals (first $0) $0))",
            "adjust": lambda xs: proportion(xs, lambda i,o: len(i) > 0) + 1 / center(xs, lambda i,o: len(i) > 1 and i.count(i[0]) == 1, factor = 2/11) + 1 / center(xs, lambda i,o: len(i) > 1 and i.count(i[0]) == 2, factor = 2/11) + 1 / center(xs, lambda i,o: len(i) > 1 and i.count(i[0]) == 3, factor = 3/11) + 1 / center(xs, lambda i,o: len(i) > 1 and i.count(i[0]) == 4, factor = 6/11),
        },
        {
            "concept": "(lambda (cut_vals (max $0) (cut_vals (min $0) $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 2) + 1 / center(xs, lambda i,o: len(i) > 1 and i.count(min(i)) > 1) + 1 / center(xs, lambda i,o: len(i) > 1 and i.count(max(i)) > 1)
        },
        {
            "concept": "(lambda (replace 2 9 $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) + 2 * proportion(xs, lambda i,o: len(i) >= 2)
        },
        {
            "concept": "(lambda (replace (first $0) (second $0) (drop 2 $0)))",
            "adjust": lambda xs: 4 * proportion(xs, lambda i,o: len(i) > 2 and 1 <= i[0] <= len(i)-2) + proportion_unique_elements(xs) + proportion_set(xs, lambda i,o: i[1] if len(i) > 1 else 0) + proportion_set(xs, lambda i,o: i[0] if len(i) > 0 else 0) + proportion_set(xs, lambda i,o: len(i)-2-i[0] if len(i) > 0 else 0) - limit(xs, 1, lambda i,o: len(i) > 0 and i[0] == 1) - limit(xs, 0, lambda i,o: len(i) > 0 and i[0] == 0) - limit(xs, 1, lambda i,o: len(i) > 1 and i[1] == len(i)-2) - limit(xs, 0, lambda i,o: len(i) > 1 and i[1] > len(i)-2)
        },
        {
            "concept": "(lambda (flatten (map (lambda (cons (/ $0 10) (singleton (% $0 10)))) $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs)
        },
        {
            "concept": "(lambda (map (lambda (if (== $0 (max $1)) (min $1) $0)) $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 1) + 1 / center(xs, lambda i,o: len(i) > 1 and i.count(max(i)) > 1, factor = 8/11) + proportion_set(xs, lambda i,o: max(0,i.count(max(i))-5) if len(i) > 0 else 0)
        },
        {
            "concept": "(lambda (map (lambda (if (or (== $0 (max $1)) (== $0 (min $1))) (- (max $1) (min $1)) $0)) $0))",
            "adjust": lambda xs: proportion_unique_elements(xs),
        },
        {
            "concept": "(lambda (map (lambda (first $1)) $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion_set(xs, lambda i,o: i[0] if len(i) > 0 else 0)
        },
        {
            "concept": "(lambda (map (lambda (- (max $0) (min $0))) (zip (droplast 1 $0) (drop 1 $0))))",
            "adjust": lambda xs: proportion_unique_elements(xs)
        },
        {
            "concept": "(lambda (flatten (mapi (lambda (lambda (cons $0 (singleton $1)))) $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs)
        },
        {
            "concept": "(lambda (flatten (map (range 1 1) $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: all(0 < ie <= 5 for ie in i)) + proportion_set(xs, lambda i,o: len(i)) + proportion(xs, lambda i,o: len(set(i)) > 3) - limit(xs, 3, lambda i,o: len(i) < 3 or len(i) > 5)
        },
        {
            "concept": "(lambda (map (lambda (* $0 (first $1))) (drop 1 $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion_set(xs, lambda i,o: i[0] if len(i) > 0 else 0) + proportion(xs, lambda i,o: i[0] <= 10 if len(i) > 0 else 0) - limit(xs, 1, lambda i,o: len(i) > 0 and i[0] == 0)
        },
        {
            "concept": "(lambda (flatten (map (lambda (if (> $0 (first $1)) (range (first $1) 1 $0) (singleton $0))) $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion_set(xs, lambda i,o: i[0] if len(i) > 0 else 0) + proportion(xs, lambda i,o: len(i) > 2)
        },
        {
            "concept": "(lambda (flatten (map (lambda (repeat $0 $0)) $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: all(ie <= 5 for ie in i)) + proportion(xs, lambda i,o: len(set(i)) > 3) + proportion(xs, lambda i,o: i.count(0) < 2) + proportion_set(xs, lambda i,o: len(i)) - limit(xs, 3, lambda i,o: len(i) < 3 or len(i) > 5)
        },
        {
            "concept": "(lambda (map (lambda (* (/ $0 10) (% $0 10))) $0))",
            "adjust": lambda xs: proportion_unique_elements(xs)
        },
        {
            "concept": "(lambda (flatten (map (lambda (append (take 1 $0) (length $0))) (group (lambda $0) $0))))",
            "adjust": lambda xs: proportion_set(xs, lambda i,o: tuple(o[1::2])) + sum(oe in [2,3,4] for i,o in xs for oe in o[1::2])/sum(len(o[1::2]) for i,o in xs)
        },
        {
            "concept": "(lambda (map (lambda (if (is_even $0) (* 3 $0) $0)) $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 5)
        },
        {
            "concept": "(lambda (mapi (lambda (lambda (* $0 $1))) $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 5) + proportion(xs, lambda i,o: all(ie <= 10 for ie in i))
        },
        {
            "concept": "(lambda (mapi (lambda (lambda (+ $0 $1))) (reverse $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 5)
        },
        {
            "concept": "(lambda (flatten (map (lambda (cons $0 (singleton (% $0 2)))) $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 5)
        },
        {
            "concept": "(lambda (mapi (lambda (lambda (if (== $0 $1) 1 0))) $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 5) + proportion(xs, lambda i,o: all(1 <= ie <= 10 for ie in i)) + proportion(xs, lambda i,o: sum(o) > 2) - limit(xs, 1, lambda i,o: sum(o) == 0)
        },
        {
            "concept": "(lambda (map (lambda (count (lambda (== $1 $0)) $1)) (range 1 1 (max $0))))",
            "adjust": lambda xs: proportion(xs, lambda i,o: len(i) > 8) + proportion(xs, lambda i,o: all(1 <= ie <= 10 for ie in i)) - limit(xs, 1, lambda i,o: sum(oe > 0 for oe in o) < 2) + proportion(xs, lambda i,o: sum(oe > 1 for oe in o) in [3,4])
        },
        {
            "concept": "(lambda (map (lambda (- 99 $0)) $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 5)
        },
        {
            "concept": "(lambda (mapi (lambda (lambda (+ $0 (- (length $2) $1)))) $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 5)
        },
        {
            "concept": "(lambda (map (lambda (+ 7 (* 3 $0))) $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 5) + proportion(xs, lambda i,o: all(0 <= ie <= 20 for ie in i))
        },
        {
            "concept": "(lambda (map (lambda (- (* $0 2) 10)) $0))",
            "adjust": lambda xs: 4 * proportion_unique_elements(xs) + 2 * proportion(xs, lambda i,o: len(i) in [4,5]) - 2 * limit(xs, 2, lambda i,o: len(i) <= 3)
        },
        {
            "concept": "(lambda (map (lambda (+ (/ $0 4) 5)) $0))",
            "adjust": lambda xs: 2 * proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 5) + proportion(xs, lambda i,o: all(10 <= ie <= 40 for ie in i))
        },
        {
            "concept": "(lambda (filter is_even (reverse $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 5)
        },
        {
            "concept": "(lambda (sort (lambda (+ (% $0 10) (/ $0 10))) (unique $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 5)
        },
        {
            "concept": "(lambda (filter (lambda (== (% $0 3) 0)) $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 5)
        },
        {
            "concept": "(lambda (cut_val (length $0) (range 1 1 10)))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion_set(xs, lambda i,o: len(i)) - limit(xs, 0, lambda i,o: len(i) == 0)
        },
        {
            "concept": "(lambda (singleton (max (cut_vals (max $0) $0))))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 5)
        },
        {
            "concept": "(lambda (cons (first $0) (singleton (last $0))))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 5)
        },
        {
            "concept": "(lambda (drop 1 (fold (lambda (lambda (append $1 (+ (last $1) $0)))) (singleton 0) $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 5) + proportion(xs, lambda i,o: all(0 <= ie <= 20 for ie in i))
        },
        {
            "concept": "(lambda (drop 1 (fold (lambda (lambda (append $1 (* (last $1) $0)))) (singleton 1) $0)))",
            "adjust": lambda xs: 2 * proportion_unique_elements(xs) + 2 * proportion(xs, lambda i,o: len(i) > 5) + proportion(xs, lambda i,o: all(2 <= ie <= 9 for ie in i)) - forbid(xs, lambda i,o: 0 in i or 0 in o) - limit(xs, 1, lambda i,o: len(i) < 3)
        },
        {
            "concept": "(lambda (mapi (lambda (lambda (max (take $1 $2)))) $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 5) + proportion(xs, lambda i,o: len(set(o))/max(1,len(o)))
        },
        {
            "concept": "(lambda (take (length (unique $0)) $0))",
            "adjust": lambda xs: proportion(xs, lambda i,o: len(i) > 5)
        },
        {
            "concept": "(lambda (fold (lambda (lambda (if (> $0 (last $1)) (append $1 $0) $1))) (take 1 $0) (drop 1 $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 5) + proportion(xs, lambda i,o: len(o) > 2)
        },
        {
            "concept": "(lambda (map (lambda (sum $0)) (zip (droplast 1 $0) (drop 1 $0))))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 5)
        },
        {
            "concept": "(lambda (flatten (zip $0 (reverse $0))))",
            "adjust": lambda xs: proportion_unique_elements(xs) - limit(xs, 2, lambda i,o: len(i) < 3 or len(i) > 6)
        },
        {
            "concept": "(lambda (map first (filter (lambda (is_even (second $0))) (zip (droplast 1 $0) (drop 1 $0)))))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 5)
        },
        {
            "concept": "(lambda (fold (lambda (lambda (append (reverse $1) $0))) empty (reverse (sort (lambda $0) $0))))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 5)
        },
        {
            "concept": "(lambda (fold (lambda (lambda (append (reverse $1) $0))) empty (sort (lambda $0) $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 5)
        },
        {
            "concept": "(lambda (flatten (zip (filteri (lambda (lambda (is_odd $1))) $0) (reverse (filteri (lambda (lambda (is_even $1))) $0)))))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 5) + proportion(xs, lambda i,o: len(i) % 2 == 0)
        },
        {
            "concept": "(lambda (filteri (lambda (lambda (== (% $1 3) 0))) $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) >= 9)
        },
        {
            "concept": "(lambda (find (== (first $0)) (drop 1 $0)))",
            "adjust": lambda xs: proportion(xs, lambda i,o: len(i) > 8) + proportion(xs, lambda i,o: all(1 <= ie <= 10 for ie in i)) + proportion_set(xs, lambda i,o: len(set(i))) - 2 * limit(xs, 1, lambda i,o: len(o) <= 1) - limit(xs, 1, lambda i,o: len(o) > 5)
        },
        {
            "concept": "(lambda (filteri (lambda (lambda (and (is_even $1) (is_odd $0)))) $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 5)
        },
        {
            "concept": "(lambda (cons (first $0) (cons (sum (drop 1 (droplast 1 $0))) (takelast 1 $0))))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 5)
        },
        {
            "concept": "(lambda (filter (lambda (> $0 (first $1))) $0))",
            "adjust": lambda xs: proportion(xs, lambda i,o: len(i) > 8) - 2 * limit(xs, 1, lambda i,o: len(o) < 2) - limit(xs, 1, lambda i,o: len(o) > 5) + proportion_unique_elements(xs)
        },
        {
            "concept": "(lambda (concat $0 (cons 0 $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) <= 5)
        },
        {
            "concept": "(lambda (map (lambda (if (== (% $0 3) 0) 1 0)) $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: 2 <= sum(o) <= 6)
        },
        {
            "concept": "(lambda (range (min $0) 1 (max $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(o) < 8) + proportion(xs, lambda i,o: len(i) > 3) + proportion_set(xs, lambda i,o: min(i) if len(i) > 0 else 0) + proportion_set(xs, lambda i,o: max(i) if len(i) > 0 else 0) - limit(xs, 1, lambda i,o: len(o) <= 1)
        },
        {
            "concept": "(lambda (range (first $0) 2 (last $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: 2 < len(o) < 8) + proportion(xs, lambda i,o: len(i) > 0 and i[0] % 2 == i[-1] % 2)
        },
        {
            "concept": "(lambda (flatten (map (lambda (repeat $0 (/ $0 10))) $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs)
        },
        {
            "concept": "(lambda (map (lambda (/ $0 10)) $0))",
            "adjust": lambda xs: proportion_unique_elements(xs)
        },
        {
            "concept": "(lambda (drop 1 (droplast 1 (sort (lambda $0) $0))))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(o) > 6)
        },
        {
            "concept": "(lambda (cons (length $0) (append (reverse $0) (length $0))))",
            "adjust": lambda xs: proportion_unique_elements(xs)
        },
        {
            "concept": "(lambda (cons (first $0) (cons 23 (cons 68 (cons 42 (cons 99 (cons 71 (singleton (last $0)))))))))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 2)
        },
        {
            "concept": "(lambda (concat (cons 17 (cons 38 (singleton 82))) (concat $0 (cons 1 (cons 55 (singleton 27))))))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: 5 >= len(i))
        },
        {
            "concept": "(lambda (map (lambda (count (== $0) $1)) $0))",
            "adjust": lambda xs: proportion_set(xs, lambda i,o: tuple(sorted(o)))
        },
        {
            "concept": "(lambda (reverse (sort (lambda $0) (unique $0))))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 5) + proportion_set(xs, lambda i,o: len(o)) + proportion_set(xs, lambda i,o: len(i) - len(o))
        },
        {
            "concept": "(lambda (flatten (zip (range 1 1 (length $0)) (sort (lambda $0) $0))))",
            "adjust": lambda xs: proportion_unique_elements(xs)
        },
        {
            "concept": "(lambda (sort (lambda $0) (map (lambda (/ $0 10)) $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(set(o))/max(1,len(o))) + proportion(xs, lambda i,o: len(i) >= 3)
        },
        {
            "concept": "(lambda (concat (filter (lambda (< (first $1) $0)) $0) (filter (lambda (> (first $1) $0)) $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: -2 < sum(ie > i[0] for ie in i)-sum(ie < i[0] for ie in i) < 2)
        },
        {
            "concept": "(lambda (find is_even $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: 5 >= len(o) >= 2)
        },
        {
            "concept": "(lambda (mapi (lambda (lambda (* (min $2) $1))) $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion_set(xs, lambda i,o: min(i) if len(i) > 0 else 0) - forbid(xs, lambda i,o: len(i) > 0 and min(i) == 0)
        },
        {
            "concept": "(lambda (map first (filter (lambda (== (second $0) 0)) (zip (droplast 1 $0) (drop 1 $0)))))",
            "adjust": lambda xs: proportion_unique_elements(xs) + 2 * proportion(xs, lambda i,o: 5 >= len(o) >= 2)
        },
        {
            "concept": "(lambda (singleton (product (filter (lambda (== (% $0 4) 0)) $0))))",
            "adjust": lambda xs: proportion_unique_elements(xs) + 2 * proportion(xs, lambda i,o: 5 >= sum(ie % 4 == 0 for ie in i) >= 2) - limit(xs, 1, lambda i,o: o[0] == 0) + proportion_set(xs, lambda i,o: o[0]) + proportion(xs, lambda i,o: len(i) >= 7)
        },
        {
            "concept": "(lambda (filter (lambda (and (> (max (take 2 $1)) $0) (> $0 (min (take 2 $1))))) $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) + 2 * proportion(xs, lambda i,o: 6 >= len(o) >= 3) + proportion_set(xs, lambda i,o: min(i[:2]) if len(i) > 1 else 0) + proportion_set(xs, lambda i,o: max(i[:2]) if len(i) > 1 else 0)
        },
        {
            "concept": "(lambda (map sum (zip $0 (reverse $0))))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: 6 >= len(o) >= 3)
        },
        {
            "concept": "(lambda (takelast (last $0) $0))",
            "adjust": lambda xs: 4 * proportion(xs, lambda i,o: len(i) > 1 and i[-1] <= len(i)-1) + proportion_unique_elements(xs) + proportion_set(xs, lambda i,o: len(o)) + proportion_set(xs, lambda i,o: len(i)-len(o)) - limit(xs, 1, lambda i,o: len(i) > 0 and i[-1] == len(i) - 1) - limit(xs, 1, lambda i,o: len(i) > 0 and i[-1] == 0)
        },
        {
            "concept": "(lambda (insert (+ (max $0) (min $0)) 3 (sort (lambda $0) $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: 7 >= len(o) >= 4)
        },
        {
            "concept": "(lambda (insert (last $0) (first $0) (unique $0)))",
            "adjust": lambda xs: 4 * proportion(xs, lambda i,o: len(i) > 1 and i[0] <= len(set(i))-1) + proportion_set(xs, lambda i,o: len(o)) + proportion_set(xs, lambda i,o: len(i)-len(o)) - limit(xs, 1, lambda i,o: len(i) > 0 and i[0] == len(set(i))) - limit(xs, 1, lambda i,o: len(i) > 0 and i[0] == 1) - forbid(xs, lambda i,o: len(i) > 0 and i[0] == 0) + proportion_unique_elements(xs),
        },
        {
            "concept": "(lambda (splice (slice 4 5 $0) (- (length $0) 2) (reverse $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: i != o) + proportion(xs, lambda i,o: 8 >= len(i) > 5)
        },
        {
            "concept": "(lambda (splice (cons 3 (cons 3 (singleton 3))) 3 $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: 7 >= len(i) >= 3) 
        },
        {
            "concept": "(lambda (take 3 (sort (lambda $0) $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) >= 3)
        },
        {
            "concept": "(lambda (cut_idx (first $0) (drop 1 $0)))",
            "adjust": lambda xs: 4 * proportion(xs, lambda i,o: len(i) > 1 and i[0] <= len(i)-1) + proportion_set(xs, lambda i,o: len(i)-i[0] if len(i) > 0 else 0) + proportion_set(xs, lambda i,o: i[0] if len(i) > 0 else 0) - limit(xs, 1, lambda i,o: len(i) > 0 and i[0] == len(i)-1) - limit(xs, 1, lambda i,o: len(i) > 0 and i[0] == 1) - forbid(xs, lambda i,o: len(i) > 0 and i[0] == 0) + proportion_unique_elements(xs),
        },
        {
            "concept": "(lambda (replace (first $0) (length $0) (drop 1 $0)))",
            "adjust": lambda xs: 4 * proportion(xs, lambda i,o: len(i) > 1 and i[0] <= len(i)-1) + proportion_set(xs, lambda i,o: len(i)) + proportion_set(xs, lambda i,o: i[0] if len(i) > 0 else 0) - limit(xs, 1, lambda i,o: len(i) > 0 and i[0] == len(i)-1) - limit(xs, 1, lambda i,o: len(i) > 0 and i[0] == 1) - forbid(xs, lambda i,o: len(i) > 0 and i[0] == 0) + proportion_unique_elements(xs),
        },
        {
            "concept": "(lambda (sort (lambda (/ $0 10)) $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: 8 > len(i) > 3)
        },
        {
            "concept": "(lambda (sort (lambda (% $0 10)) $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: 8 > len(i) > 3)
        },
        {
            "concept": "(lambda (filter (lambda (== $0 (first $1))) (drop 1 $0)))",
            "adjust": lambda xs: proportion(xs, lambda i,o: len(i) >= 8) + proportion_set(xs, lambda i,o: len(o)) - limit(xs, 2, lambda i,o: len(o) < 2),
        },
        {
            "concept": "(lambda (reverse (filteri (lambda (lambda (is_odd $1))) (reverse $0))))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) >= 7),
        },
        {
            "concept": "(lambda (map (lambda (* $0 (if (is_even (length $1)) 2 3))) $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) - forbid(xs, lambda i,o: len(i) == 0) - limit(xs, 1, lambda i,o: len(i) == 1) + 2 / center(xs, lambda i,o: len(i) % 2 == 0)
        },
        {
            "concept": "(lambda (singleton (sum (filter is_even $0))))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) >= 7) + proportion(xs, lambda i,o: 5 >= len(o) >= 2)
        },
        {
            "concept": "(lambda (map (lambda (length $1)) $0))",
            "adjust": lambda xs: proportion_unique_elements(xs)
        },
        {
            "concept": "(lambda (map (lambda (+ (* (% $0 10) 10) (/ $0 10))) $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) >= 3)
        },
        {
            "concept": "(lambda (fold (lambda (lambda (cons $0 (reverse $1)))) empty $0))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) >= 4)
        },
        {
            "concept": "(lambda (drop 2 (droplast 2 $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 4)
        },
        {
            "concept": "(lambda (drop (first $0) (droplast (last $0) $0)))",
            "adjust": lambda xs: 4 * proportion(xs, lambda i,o: len(i) > 1 and (i[0] + i[-1]) <= len(i)) + proportion_set(xs, lambda i,o: i[0] if len(i) > 0 else 0) + proportion_set(xs, lambda i,o: i[-1] if len(i) > 0 else 0) + proportion_set(xs, lambda i,o: len(o)) - limit(xs, 2, lambda i,o: len(i) > 0 and i[0] == 0 or i[-1] == 0) + proportion_unique_elements(xs)
        },
        {
            "concept": "(lambda (unique (flatten (zip $0 (reverse $0)))))",
            "adjust": lambda xs: proportion_unique_elements(xs) + 2 * proportion_set(xs, lambda i,o: len(i)-len(o)) + 2 * proportion_set(xs, lambda i,o: len(o))
        },
        {
            "concept": "(lambda (mapi (lambda (lambda (count (== $0) (take $1 $2)))) $0))",
            "adjust": lambda xs: proportion(xs, lambda i,o: sum(oe in [2,3,4] for oe in o)/max(1,len(o)))
        },
        {
            "concept": "(lambda (take (first $0) (reverse $0)))",
            "adjust": lambda xs: 4 * proportion(xs, lambda i,o: len(i) > 1 and i[0] <= len(i)) + proportion_unique_elements(xs) + proportion_set(xs, lambda i,o: len(o)) + proportion_set(xs, lambda i,o: len(i)-len(o)) - limit(xs, 1, lambda i,o: len(i) > 0 and i[0] == len(i)) - limit(xs, 1, lambda i,o: len(i) > 0 and i[0] == 0)
        },
        {
            "concept": "(lambda (range (min $0) 2 (max $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: 2 < len(o) < 8) + proportion_set(xs, lambda i,o: min(i) if len(i) > 0 else 0) + proportion_set(xs, lambda i,o: max(i) if len(i) > 0 else 0)
        },
        {
            "concept": "(lambda (sort (lambda $0) (map length (group (lambda $0) $0))))",
            "adjust": lambda xs: proportion_set(xs, lambda i,o: tuple(sorted(o))) + proportion(xs, lambda i,o: len(i) >= 3) + proportion_set(xs, lambda i,o: len(i)-len(o)) + proportion_set(xs, lambda i,o: len(o)) - limit(xs, 1, lambda i,o: sum(o) == len(o))
        },
        {
            "concept": "(lambda (singleton (/ (sum $0) (length $0))))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: 2 < len(i) < 6) + proportion_set(xs, lambda i,o: o[0])
        },
        {
            "concept": "(lambda (map length (group (lambda $0) $0)))",
            "adjust": lambda xs: proportion_set(xs, lambda i,o: tuple(sorted(o))) + proportion(xs, lambda i,o: len(i) >= 3) + proportion_set(xs, lambda i,o: len(i)-len(o)) + proportion_set(xs, lambda i,o: len(o)) - limit(xs, 1, lambda i,o: sum(o) == len(o))
        },
        {
            "concept": "(lambda (flatten (map (lambda (drop 1 $0)) (group (lambda $0) $0))))",
            "adjust": lambda xs: proportion_set(xs, lambda i,o: tuple(sorted(o))) + proportion(xs, lambda i,o: len(i) >= 3) + proportion_set(xs, lambda i,o: len(i)-len(o)) + proportion_set(xs, lambda i,o: len(o))
        },
        {
            "concept": "(lambda (fold (lambda (lambda (concat $1 (drop 1 (range (last $1) (if (> $0 (last $1)) 1 -1) $0))))) (take 1 $0) (drop 1 $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs) - forbid(xs, lambda i,o: len(o) <= 1) - limit(xs, 1, lambda i,o: len(i) <= 2)
        },
        {
            "concept": "(lambda (map (lambda (/ $0 2)) (filter is_even $0)))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) >= 7)
        },
        {
            "concept": "(lambda (fold (lambda (lambda (append $1 (+ (last $1) $0)))) (take 1 (unique $0)) (drop 1 (unique $0))))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 5) + proportion(xs, lambda i,o: all(0 <= ie <= 20 for ie in i)) + proportion(xs, lambda i,o: 5 >= len(i)-len(set(i)) >= 2) - limit(xs, 1, lambda i,o: len(o) <= 1) + proportion_set(xs, lambda i,o: len(o))
        },
        {
            "concept": "(lambda (filter (lambda (== 1 (count (== $0) $1))) $0))",
            "adjust": lambda xs: proportion(xs, lambda i,o: len(i) > 5) + proportion(xs, lambda i,o: all(0 <= ie <= 20 for ie in i)) + proportion_set(xs, lambda i,o: len(i)-len(set(i))) + proportion(xs, lambda i,o: 2 <= len(o) <= 5)
        },
        {
            "concept": "(lambda (singleton (- (length $0) (length (unique $0)))))",
            "adjust": lambda xs: 2 * proportion_set(xs, lambda i,o: o[0]) + proportion_unique_elements(xs)
        },
        {
            "concept": "(lambda (singleton (count (lambda ((== (length $1)) $0)) $0)))",
            "adjust": lambda xs: 3 * proportion_set(xs, lambda i,o: min(10,o[0]+5)) + proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) >= 7) - limit(xs, 2, lambda i,o: o[0] <= 1)
        },
        {
            "concept": "(lambda (singleton (count is_even $0)))",
            "adjust": lambda xs: 2 * proportion_set(xs, lambda i,o: o[0]) + proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) >= 7)
        },
        {
            "concept": "(lambda (fold (lambda (lambda (append (reverse $1) $0))) empty (reverse (unique (sort (lambda $0) $0)))))",
            "adjust": lambda xs: proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) > 5) + proportion_set(xs, lambda i,o: len(i)-len(o)) + proportion_set(xs, lambda i,o: len(o))
        },
        {
            "concept": "(lambda (singleton (count is_odd $0)))",
            "adjust": lambda xs: 2 * proportion_set(xs, lambda i,o: o[0]) + proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) >= 7)
        },
        {
            "concept": "(lambda (singleton (count (lambda (== 3 $0)) $0)))",
            "adjust": lambda xs: 2 * proportion_set(xs, lambda i,o: o[0]) + proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) >= 7)
        },
        {
            "concept": "(lambda (singleton (count (lambda (== (first $1) $0)) (drop 1 $0))))",
            "adjust": lambda xs: 2 * proportion_set(xs, lambda i,o: o[0]) + proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) >= 7)
        },
        {
            "concept": "(lambda (singleton (length (unique $0))))",
            "adjust": lambda xs: 2 * proportion_set(xs, lambda i,o: o[0]) + proportion_unique_elements(xs) + proportion(xs, lambda i,o: len(i) >= 7)
        },
        {
            "concept": "(lambda (first (reverse (fold (lambda (lambda (if (== $0 0) (cons empty $1) (cons (append (first $1) $0) (drop 1 $1))))) (singleton empty) $0))))",
            "adjust": lambda xs: proportion(xs, lambda i,o: len(i) >= 9) + 4 * proportion(xs, lambda i,o: 4 > len(words(i)) > 2) - limit(xs, 2, lambda i,o: [] in words(i)) + proportion_set(xs, lambda i,o: tuple(o)) + 2 * proportion(xs, lambda i,o: 1 <= len(o) <= 4)
        },
        {
            "concept": "(lambda (first (fold (lambda (lambda (if (== $0 0) (cons empty $1) (cons (append (first $1) $0) (drop 1 $1))))) (singleton empty) $0)))",
            "adjust": lambda xs: proportion(xs, lambda i,o: len(i) >= 9) + 4 * proportion(xs, lambda i,o: 4 > len(words(i)) > 2) - limit(xs, 2, lambda i,o: [] in words(i)) + proportion_set(xs, lambda i,o: tuple(o)) + 2 * proportion(xs, lambda i,o: 1 <= len(o) <= 4)
        },
        {
            "concept": "(lambda (map first (reverse (fold (lambda (lambda (if (== $0 0) (cons empty $1) (cons (append (first $1) $0) (drop 1 $1))))) (singleton empty) $0))))",
            "adjust": lambda xs: proportion(xs, lambda i,o: len(i) >= 9) + 4 * proportion(xs, lambda i,o: 4 > len(words(i)) > 2) - limit(xs, 2, lambda i,o: [] in words(i)) + proportion_set(xs, lambda i,o: tuple(o)) + 2 * proportion(xs, lambda i,o: 3 <= len(o) <= 4)
        },
        {
            "concept": "(lambda (flatten (map reverse (reverse (fold (lambda (lambda (if (== $0 0) (cons empty $1) (cons (append (first $1) $0) (drop 1 $1))))) (singleton empty) $0)))))",
            "adjust": lambda xs: proportion(xs, lambda i,o: len(i) >= 9) + 4 * proportion(xs, lambda i,o: 4 > len(words(i)) > 2) - limit(xs, 2, lambda i,o: [] in words(i)) + proportion_set(xs, lambda i,o: tuple(o)) + proportion(xs, lambda i,o: [ie for ie in i if ie != 0] != o)
        },
    ]

def words(xs, sep=0):
    words = []
    word = []
    looped = False
    for x in xs:
        looped = True
        if x == sep:
            words.append(word)
            word = []
            looped = False
        else:
            word.append(x)
    if looped:
        words.append(word)
    return words


def model_comparison_wave_3():
    return [
        {'concept': '(lambda (singleton (third $0)))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 3 for i, o in xs) else 0) + 2 * proportion_unique_elements(xs)
        }
        ,
        {'concept': '(lambda (if (> 3 (length $0)) empty (singleton (third $0))))',
         'adjust': lambda xs: 6/center(xs, lambda i,o: len(i) >= 3) + 2 * proportion_unique_elements(xs)
        },
        {'concept': '(lambda (singleton (nth 7 $0)))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 7 for i, o in xs) else 0) + 2 * proportion_unique_elements(xs)
        },
        {'concept': '(lambda (if (> 7 (length $0)) empty (singleton (nth 7 $0))))',
         'adjust': lambda xs: (3.0 if 0.6 >= sum(len(i) >= 7 for i, o in xs)/len(xs) >= 0.4 else 0) + 2 * proportion_unique_elements(xs), 
        },
        {'concept': '(lambda (singleton (nth (first $0) (drop 1 $0))))',
         'adjust': lambda xs: 2.0 * proportion(xs, lambda i,o: i[0] <= len(i)-1) + 2.0 * proportion_set(xs, lambda i,o: i[0]) + 2 * proportion_unique_elements(xs) + 2 * proportion_set(xs, lambda i,o: len(i)-i[0]) - 0.5 * limit(xs, 1, lambda i,o: i[0] == 1)
        },
        {'concept': '(lambda (take 2 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 2 for i, o in xs) else 0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (take 2 $0))',
         'adjust': lambda xs: (3.0 if 0.6 >= sum(len(i) >= 2 for i, o in xs)/len(xs) >= 0.4 else 0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (take 6 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 6 for i, o in xs) else 0) + 2.0 * proportion_unique_elements(xs)
        },
        {'concept': '(lambda (take 6 $0))',
         'adjust': lambda xs: (3.0 if 0.6 >= sum(len(i) >= 6 for i, o in xs)/len(xs) >= 0.4 else 0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (take (first $0) (drop 1 $0)))',
         'adjust': lambda xs: (3.0 if all(i[0] <= len(i)-1 for i, o in xs) else 0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (slice 2 4 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 4 for i, o in xs) else 0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (slice 2 4 $0))',
         'adjust': lambda xs: (sum(2 > len(i) for i,o in xs) >= 2) + (sum(4 > len(i) >= 2 for i,o in xs) >= 2) + (sum(len(i) >= 4 for i,o in xs) >= 4) + (2.0 * proportion_unique_elements(xs)),
        },
        {'concept': '(lambda (slice 3 7 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 7 for i, o in xs) else 0) + (2.0 * proportion_unique_elements(xs)),
        },
        {'concept': '(lambda (slice 3 7 $0))',
         'adjust': lambda xs: (sum(3 > len(i) for i,o in xs) >= 2) + (sum(7 > len(i) >= 3 for i,o in xs) >= 2) + (sum(len(i) >= 7 for i,o in xs) >= 4) + (2.0 * proportion_unique_elements(xs))
        },
        {'concept': '(lambda (slice (first $0) (second $0) (drop 2 $0)))',
         'adjust': lambda xs: 4.0 * proportion(xs, lambda i,o: len(i)-2 >= i[1] >= i[0] > 0) + proportion_set(xs, lambda i,o: i[0]) + proportion_set(xs, lambda i,o: i[1]) + proportion_set(xs, lambda i,o: len(i)-i[1])  - 0.5 * limit(xs, 1, lambda i,o: len(i)-2 == i[1]) - 0.5 * limit(xs, 1, lambda i,o: i[1] == i[0]) + 2.0 * proportion_unique_elements(xs)
        },
        {'concept': '(lambda (replace 2 8 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 2 for i, o in xs) else 0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (replace 2 8 $0))',
         'adjust': lambda xs: (3.0 if 0.6 >= sum(2 > len(i) for i, o in xs)/len(xs) >= 0.4 else 0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (replace 6 3 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 6 for i, o in xs) else 0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (replace 6 3 $0))',
         'adjust': lambda xs: (3.0 if 0.6 >= sum(6 > len(i) for i, o in xs)/len(xs) >= 0.4 else 0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (replace 1 (last $0) $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 1 for i, o in xs) else 0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (insert 8 2 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 2 for i, o in xs) else 0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (insert 5 2 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 2 for i, o in xs) else 0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (insert (if (> 5 (length $0)) 8 5) 2 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 2 for i, o in xs) else 0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (insert (if (> 5 (first $0)) 8 5) 2 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 2 for i, o in xs) else 0) + 2.0 * proportion_unique_elements(xs) + 4.0 / center(xs, lambda i,o: i[0] > 5),
        },
        {'concept': '(lambda (cut_idx 2 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 2 for i, o in xs) else 0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (cut_idx 3 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 3 for i, o in xs) else 0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (cut_idx (if (== (first $0) (second $0)) 2 3) $0))',
         'adjust': lambda xs: 4 * proportion(xs, lambda i,o: len(i) >= 3 and i[1] != i[2]) + 2 / center(xs, lambda i,o: len(i) >= 3 and i[0] == i[1]) + 2 * proportion_set(xs, lambda i,o: (i[0], i[1]) if len(i) > 1 else (0, 0)) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (cut_idx (if (> (first $0) (second $0)) 2 3) $0))',
         'adjust': lambda xs: 4 * proportion(xs, lambda i,o: len(i) >= 3 and ((i[0] > i[1] and i[0] < i[2]) or (i[0] > i[2] and i[0] < i[1])))  + 2 / center(xs, lambda i,o: len(i) >= 3 and i[0] > i[1]) + 2 * proportion_set(xs, lambda i,o: (i[0], i[1]) if len(i) > 1 else (0, 0)) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (drop 2 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 2 for i, o in xs) else 0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (droplast 2 $0))',
         'adjust': lambda xs: 4 * proportion(xs, lambda i,o: len(i) >= 2) + proportion_set(xs, lambda i,o: len(o)) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda ((if (== (first $0) (second $0)) drop droplast) 2 $0))',
         'adjust': lambda xs: 4 * proportion(xs, lambda i,o: len(i) >= 3 and ((i[0]==i[1] and i[-1] != i[-2]) or (i[0]!=i[1] and i[-1] == i[-2]))) + 2.0 * proportion_unique_elements(xs)
        },
        {'concept': '(lambda ((if (> (first $0) (last $0)) drop droplast) 2 $0))',
         'adjust': lambda xs: 4 * proportion(xs, lambda i,o: len(i) >= 3 and (i[0]!=i[-1])) + 2 / center(xs, lambda i,o: len(i) >= 3 and i[0] > i[-1]) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (swap 1 4 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 4 for i, o in xs) else 0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (swap 2 3 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 3 for i, o in xs) else 0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (if (== (second $0) (third $0)) (swap 1 4 $0) (swap 2 3 $0)))',
         'adjust': lambda xs: 4 * proportion(xs, lambda i,o: len(i) >= 4 and ((i[1] == i[2] and i[0] != i[3]) or (i[1] != i[2] and i[0] == i[3]))) + 4 / center(xs, lambda i,o: i[1] == i[2]) + 2.0 * proportion_unique_elements(xs)
        },
        {'concept': '(lambda (if (> (second $0) (third $0)) (swap 2 3 $0) (swap 1 4 $0)))',
         'adjust': lambda xs: 4 * proportion(xs, lambda i,o: len(i) >= 4 and ((i[1] > i[2] and i[0] <= i[3]) or (i[1] <= i[2] and i[0] > i[3]))) + 2 / center(xs, lambda i,o: i[1] > i[2]) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (append $0 3))',
         'adjust': lambda xs: 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (append $0 9))',
         'adjust': lambda xs: 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (if (== (length $0) 3) (append $0 3) (if (== (length $0) 9) (append $0 9) $0)))',
         'adjust': lambda xs: 4 / center(xs, lambda i,o: len(i) in [3, 9], factor = 8/11) + 1 / max(1, abs(sum(len(i) == 3 for i,o in xs) - sum(len(i) == 9 for i,o in xs))) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (if (is_in $0 3) (append $0 3) (if (is_in $0 9) (append $0 9) $0)))',
         'adjust': lambda xs: 4 / center(xs, lambda i,o: (3 in i and 9 not in i), factor = 4/11) + 4 / center(xs, lambda i,o: (9 in i and 3 not in i), factor = 4/11) + 4 / center(xs, lambda i,o: (3 not in i and 9 not in i), factor = 3/11) + 2.0 * proportion_unique_elements(xs)
        },
        {'concept': '(lambda (singleton 9))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda (cons 5 (singleton 2)))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda (cons 8 (cons 2 (cons 7 (cons 0 (singleton 3))))))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda (cons 1 (cons 9 (cons 4 (cons 3 (cons 2 (cons 5 (cons 8 (cons 0 (cons 4 (singleton 9)))))))))))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda $0)',
         'adjust': lambda xs: 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (cons 7 $0))',
         'adjust': lambda xs: 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (cons 9 (cons 6 (cons 3 (cons 8 (cons 5 $0))))))',
         'adjust': lambda xs: 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (take 1 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) > 0 for i,o in xs) else 0.0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (drop 1 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) > 0 for i,o in xs) else 0.0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (cons (first $0) $0))',
         'adjust': lambda xs: 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (concat (repeat (first $0) 5) $0))',
         'adjust': lambda xs: 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (repeat (first $0) 10))',
         'adjust': lambda xs: 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (concat (repeat (first $0) 2) (drop 2 $0)))',
         'adjust': lambda xs: 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (concat (repeat (third $0) 3) (drop 3 $0)))',
         'adjust': lambda xs: 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (concat (slice 3 4 $0) (concat (take 2 $0) (drop 4 $0))))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 4 for i,o in xs) else 0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (cut_idx 5 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 5 for i,o in xs) else 0.0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (insert 4 7 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 7 for i,o in xs) else 0.0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (drop 7 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 7 for i,o in xs) else 0.0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (swap 4 8 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 8 for i,o in xs) else 0.0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (swap 3 1 (replace 4 4 (cut_idx 6 (take 7 $0)))))',
         'adjust': lambda xs: 4 * proportion(xs, lambda i,o: len(i) >= 7) + 2 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (singleton (last $0)))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 1 for i,o in xs) else 0.0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (droplast 1 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 1 for i,o in xs) else 0.0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (drop (first $0) (drop 1 $0)))',
         'adjust': lambda xs: (3.0 if all(i[0] <= len(i)-1 for i,o in xs) else 0.0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (drop 1 (droplast 1 $0)))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 2 for i,o in xs) else 0.0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (cons 9 (append $0 7)))',
         'adjust': lambda xs: 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (append (drop 1 $0) (first $0)))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 1 for i,o in xs) else 0.0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (cons (last $0) (append (drop 1 (droplast 1 $0)) (first $0))))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 2 for i,o in xs) else 0.0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (concat $0 (cons 7 (cons 3 (cons 8 (cons 4 (singleton 3)))))))',
         'adjust': lambda xs: 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (concat (cons 9 (cons 3 (cons 4 (singleton 0)))) (concat $0 (cons 7 (cons 2 (cons 9 (singleton 1)))))))',
         'adjust': lambda xs: 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (concat $0 $0))',
         'adjust': lambda xs: 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (map (lambda (+ 2 $0)) $0))',
         'adjust': lambda xs: 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (flatten (map (lambda (cons $0 (singleton $0))) $0)))',
         'adjust': lambda xs: 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (mapi + $0))',
         'adjust': lambda xs: 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (filter (lambda (> $0 7)) $0))',
         "adjust": lambda xs: min(1.0, 1.0/(len(xs)-2)*sum((len(o)/len(i) <= 0.75 if len(i) > 0 else 1) for i, o in xs)) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (filteri (lambda (lambda (is_odd $1))) $0))',
         'adjust': lambda xs: 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (cons (max $0) (cons (last $0) (cons (length $0) (cons (first $0) (singleton (min $0)))))))',
         "adjust": lambda xs: 2 * proportion_set(xs, lambda i,o: len(i)) + 2 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (singleton (length $0)))',
         'adjust': lambda xs: 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (singleton (max $0)))',
         "adjust": lambda xs: 4 * proportion_set(xs, lambda i,o: o[0]) + 2 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (singleton (sum $0)))',
         'adjust': lambda xs: 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (reverse $0))',
         'adjust': lambda xs: 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (singleton (third $0)))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 3 for i, o in xs) else 0) + 2.0 * proportion_unique_elements(xs)
        },
        {'concept': '(lambda (if (> 3 (length $0)) empty (singleton (third $0))))',
         'adjust': lambda xs: (3.0 if 0.6 >= sum(len(i) >= 3 for i, o in xs)/len(xs) >= 0.4 else 0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (singleton (nth 7 $0)))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 7 for i, o in xs) else 0) + 2.0 * proportion_unique_elements(xs)
        },
        {'concept': '(lambda (if (> 7 (length $0)) empty (singleton (nth 7 $0))))',
         'adjust': lambda xs: (3.0 if 0.6 >= sum(len(i) >= 7 for i, o in xs)/len(xs) >= 0.4 else 0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (singleton (nth (first $0) (drop 1 $0))))',
         'adjust': lambda xs: (3.0 if all(i[0] <= len(i)-1 for i, o in xs) else 0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (swap 1 4 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 4 for i, o in xs) else 0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (swap 2 3 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 3 for i, o in xs) else 0) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (if (== (second $0) (third $0)) (swap 1 4 $0) (swap 2 3 $0)))',
         'adjust': lambda xs: 4 * proportion(xs, lambda i,o: len(i) >= 4 and ((i[1] == i[2] and i[0] != i[3]) or (i[1] != i[2] and i[0] == i[3]))) + 2 / center(xs, lambda i,o: i[1] == i[2]) + 2.0 * proportion_unique_elements(xs)
        },
        {'concept': '(lambda (if (> (second $0) (third $0)) (swap 2 3 $0) (swap 1 4 $0)))',
         'adjust': lambda xs: 4 * proportion(xs, lambda i,o: len(i) >= 4 and ((i[1] > i[2] and i[0] <= i[3]) or (i[1] <= i[2] and i[0] > i[3]))) + 2 / center(xs, lambda i,o: i[1] > i[2]) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (cons 18 (cons 42 (cons 77 (cons 20 (singleton 36))))))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda (cons 81 (cons 99 (cons 41 (cons 23 (cons 22 (cons 75 (cons 68 (cons 30 (cons 24 (singleton 69)))))))))))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda (cons 92 (cons 63 (cons 34 (cons 18 (cons 55 $0))))))',
         'adjust': lambda xs: proportion_unique_elements(xs),
        },
        {'concept': '(lambda (repeat (first $0) 10))',
         'adjust': lambda xs: proportion_unique_elements(xs),
        },
        {'concept': '(lambda (concat (slice 3 4 $0) (concat (take 2 $0) (drop 4 $0))))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 4 for i,o in xs) else 0) + 2.0 * proportion_unique_elements(xs)
        },
        {'concept': '(lambda (drop 1 (droplast 1 $0)))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 2 for i,o in xs) else 0) + 2.0 * proportion_unique_elements(xs)
        },
        {'concept': '(lambda (cons 98 (append $0 37)))',
         'adjust': lambda xs: proportion_unique_elements(xs),
        },
        {'concept': '(lambda (concat (cons 11 (cons 21 (cons 43 (singleton 19)))) (concat $0 (cons 7 (cons 89 (cons 0 (singleton 57)))))))',
         'adjust': lambda xs: proportion_unique_elements(xs),
        },
        {'concept': '(lambda (mapi + $0))',
         'adjust': lambda xs: proportion_unique_elements(xs),
        },
        {'concept': '(lambda (filter (lambda (> $0 49)) $0))',
         "adjust": lambda xs: min(1.0, 1.0/(len(xs)-2)*sum((len(o)/len(i) <= 0.75 if len(i) > 0 else 1) for i, o in xs)) + 2.0 * proportion_unique_elements(xs),
        },
        {'concept': '(lambda (reverse $0))',
         'adjust': lambda xs: proportion_unique_elements(xs),
        },
    ]

def model_comparison_wave_3_long():
    return [
        {'concept': '(lambda (singleton (third $0)))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 3 for i, o in xs) else 0) 
        }
        ,
        {'concept': '(lambda (if (> 3 (length $0)) empty (singleton (third $0))))',
         'adjust': lambda xs: 6/center(xs, lambda i,o: len(i) >= 3) 
        },
        {'concept': '(lambda (singleton (nth 7 $0)))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 7 for i, o in xs) else 0) 
        },
        {'concept': '(lambda (if (> 7 (length $0)) empty (singleton (nth 7 $0))))',
         'adjust': lambda xs: (3.0 if 0.6 >= sum(len(i) >= 7 for i, o in xs)/len(xs) >= 0.4 else 0) , 
        },
        {'concept': '(lambda (singleton (nth (first $0) (drop 1 $0))))',
         'adjust': lambda xs: 2.0 * proportion(xs, lambda i,o: i[0] <= len(i)-1) + 2.0 * proportion_set(xs, lambda i,o: i[0]) + 2 * proportion_set(xs, lambda i,o: len(i)-i[0]) - 0.5 * limit(xs, 1, lambda i,o: i[0] == 1)
        },
        {'concept': '(lambda (take 2 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 2 for i, o in xs) else 0),
        },
        {'concept': '(lambda (take 2 $0))',
         'adjust': lambda xs: (3.0 if 0.6 >= sum(len(i) >= 2 for i, o in xs)/len(xs) >= 0.4 else 0),
        },
        {'concept': '(lambda (take 6 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 6 for i, o in xs) else 0)
        },
        {'concept': '(lambda (take 6 $0))',
         'adjust': lambda xs: (3.0 if 0.6 >= sum(len(i) >= 6 for i, o in xs)/len(xs) >= 0.4 else 0),
        },
        {'concept': '(lambda (take (first $0) (drop 1 $0)))',
         'adjust': lambda xs: (3.0 if all(i[0] <= len(i)-1 for i, o in xs) else 0),
        },
        {'concept': '(lambda (slice 2 4 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 4 for i, o in xs) else 0),
        },
        {'concept': '(lambda (slice 2 4 $0))',
         'adjust': lambda xs: (sum(2 > len(i) for i,o in xs) >= 2) + (sum(4 > len(i) >= 2 for i,o in xs) >= 2) + (sum(len(i) >= 4 for i,o in xs) >= 4),
        },
        {'concept': '(lambda (slice 3 7 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 7 for i, o in xs) else 0),
        },
        {'concept': '(lambda (slice 3 7 $0))',
         'adjust': lambda xs: (sum(3 > len(i) for i,o in xs) >= 2) + (sum(7 > len(i) >= 3 for i,o in xs) >= 2) + (sum(len(i) >= 7 for i,o in xs) >= 4)
        },
        {'concept': '(lambda (slice (first $0) (second $0) (drop 2 $0)))',
         'adjust': lambda xs: 4.0 * proportion(xs, lambda i,o: len(i)-2 >= i[1] >= i[0] > 0) + proportion_set(xs, lambda i,o: i[0]) + proportion_set(xs, lambda i,o: i[1]) + proportion_set(xs, lambda i,o: len(i)-i[1])  - 0.5 * limit(xs, 1, lambda i,o: len(i)-2 == i[1]) - 0.5 * limit(xs, 1, lambda i,o: i[1] == i[0])
        },
        {'concept': '(lambda (replace 2 8 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 2 for i, o in xs) else 0),
        },
        {'concept': '(lambda (replace 2 8 $0))',
         'adjust': lambda xs: (3.0 if 0.6 >= sum(2 > len(i) for i, o in xs)/len(xs) >= 0.4 else 0),
        },
        {'concept': '(lambda (replace 6 3 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 6 for i, o in xs) else 0),
        },
        {'concept': '(lambda (replace 6 3 $0))',
         'adjust': lambda xs: (3.0 if 0.6 >= sum(6 > len(i) for i, o in xs)/len(xs) >= 0.4 else 0),
        },
        {'concept': '(lambda (replace 1 (last $0) $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 1 for i, o in xs) else 0),
        },
        {'concept': '(lambda (insert 8 2 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 2 for i, o in xs) else 0),
        },
        {'concept': '(lambda (insert 5 2 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 2 for i, o in xs) else 0),
        },
        {'concept': '(lambda (insert (if (> 5 (length $0)) 8 5) 2 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 2 for i, o in xs) else 0),
        },
        {'concept': '(lambda (insert (if (> 5 (first $0)) 8 5) 2 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 2 for i, o in xs) else 0) + 4.0 / center(xs, lambda i,o: i[0] > 5),
        },
        {'concept': '(lambda (cut_idx 2 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 2 for i, o in xs) else 0),
        },
        {'concept': '(lambda (cut_idx 3 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 3 for i, o in xs) else 0),
        },
        {'concept': '(lambda (cut_idx (if (== (first $0) (second $0)) 2 3) $0))',
         'adjust': lambda xs: 4 * proportion(xs, lambda i,o: len(i) >= 3 and i[1] != i[2]) + 2 / center(xs, lambda i,o: len(i) >= 3 and i[0] == i[1]) + 2 * proportion_set(xs, lambda i,o: (i[0], i[1]) if len(i) > 1 else (0, 0)),
        },
        {'concept': '(lambda (cut_idx (if (> (first $0) (second $0)) 2 3) $0))',
         'adjust': lambda xs: 4 * proportion(xs, lambda i,o: len(i) >= 3 and ((i[0] > i[1] and i[0] < i[2]) or (i[0] > i[2] and i[0] < i[1])))  + 2 / center(xs, lambda i,o: len(i) >= 3 and i[0] > i[1]) + 2 * proportion_set(xs, lambda i,o: (i[0], i[1]) if len(i) > 1 else (0, 0)),
        },
        {'concept': '(lambda (drop 2 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 2 for i, o in xs) else 0),
        },
        {'concept': '(lambda (droplast 2 $0))',
         'adjust': lambda xs: 4 * proportion(xs, lambda i,o: len(i) >= 2) + proportion_set(xs, lambda i,o: len(o)),
        },
        {'concept': '(lambda ((if (== (first $0) (second $0)) drop droplast) 2 $0))',
         'adjust': lambda xs: 4 * proportion(xs, lambda i,o: len(i) >= 3 and ((i[0]==i[1] and i[-1] != i[-2]) or (i[0]!=i[1] and i[-1] == i[-2])))
        },
        {'concept': '(lambda ((if (> (first $0) (last $0)) drop droplast) 2 $0))',
         'adjust': lambda xs: 4 * proportion(xs, lambda i,o: len(i) >= 3 and (i[0]!=i[-1])) + 2 / center(xs, lambda i,o: len(i) >= 3 and i[0] > i[-1]),
        },
        {'concept': '(lambda (swap 1 4 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 4 for i, o in xs) else 0),
        },
        {'concept': '(lambda (swap 2 3 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 3 for i, o in xs) else 0),
        },
        {'concept': '(lambda (if (== (second $0) (third $0)) (swap 1 4 $0) (swap 2 3 $0)))',
         'adjust': lambda xs: 4 * proportion(xs, lambda i,o: len(i) >= 4 and ((i[1] == i[2] and i[0] != i[3]) or (i[1] != i[2] and i[0] == i[3]))) + 4 / center(xs, lambda i,o: i[1] == i[2])
        },
        {'concept': '(lambda (if (> (second $0) (third $0)) (swap 2 3 $0) (swap 1 4 $0)))',
         'adjust': lambda xs: 4 * proportion(xs, lambda i,o: len(i) >= 4 and ((i[1] > i[2] and i[0] <= i[3]) or (i[1] <= i[2] and i[0] > i[3]))) + 2 / center(xs, lambda i,o: i[1] > i[2]),
        },
        {'concept': '(lambda (append $0 3))',
         'adjust': lambda xs: 1.0
        },
        {'concept': '(lambda (append $0 9))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda (if (== (length $0) 3) (append $0 3) (if (== (length $0) 9) (append $0 9) $0)))',
         'adjust': lambda xs: 4 / center(xs, lambda i,o: len(i) in [3, 9], factor = 8/11) + 1 / max(1, abs(sum(len(i) == 3 for i,o in xs) - sum(len(i) == 9 for i,o in xs))),
        },
        {'concept': '(lambda (if (is_in $0 3) (append $0 3) (if (is_in $0 9) (append $0 9) $0)))',
         'adjust': lambda xs: 4 / center(xs, lambda i,o: (3 in i and 9 not in i), factor = 4/11) + 4 / center(xs, lambda i,o: (9 in i and 3 not in i), factor = 4/11) + 4 / center(xs, lambda i,o: (3 not in i and 9 not in i), factor = 3/11)
        },
        {'concept': '(lambda (singleton 9))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda (cons 5 (singleton 2)))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda (cons 8 (cons 2 (cons 7 (cons 0 (singleton 3))))))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda (cons 1 (cons 9 (cons 4 (cons 3 (cons 2 (cons 5 (cons 8 (cons 0 (cons 4 (singleton 9)))))))))))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda $0)',
         'adjust': lambda xs: 1.0
        },
        {'concept': '(lambda (cons 7 $0))',
         'adjust': lambda xs: 1.0
        },
        {'concept': '(lambda (cons 9 (cons 6 (cons 3 (cons 8 (cons 5 $0))))))',
         'adjust': lambda xs: 1.0
        },
        {'concept': '(lambda (take 1 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) > 0 for i,o in xs) else 0.0),
        },
        {'concept': '(lambda (drop 1 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) > 0 for i,o in xs) else 0.0),
        },
        {'concept': '(lambda (cons (first $0) $0))',
         'adjust': lambda xs: 1.0
        },
        {'concept': '(lambda (concat (repeat (first $0) 5) $0))',
         'adjust': lambda xs: 1.0
        },
        {'concept': '(lambda (repeat (first $0) 10))',
         'adjust': lambda xs: 1.0
        },
        {'concept': '(lambda (concat (repeat (first $0) 2) (drop 2 $0)))',
         'adjust': lambda xs: 1.0
        },
        {'concept': '(lambda (concat (repeat (third $0) 3) (drop 3 $0)))',
         'adjust': lambda xs: 1.0
        },
        {'concept': '(lambda (concat (slice 3 4 $0) (concat (take 2 $0) (drop 4 $0))))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 4 for i,o in xs) else 0),
        },
        {'concept': '(lambda (cut_idx 5 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 5 for i,o in xs) else 0.0),
        },
        {'concept': '(lambda (insert 4 7 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 7 for i,o in xs) else 0.0),
        },
        {'concept': '(lambda (drop 7 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 7 for i,o in xs) else 0.0),
        },
        {'concept': '(lambda (swap 4 8 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 8 for i,o in xs) else 0.0),
        },
        {'concept': '(lambda (swap 3 1 (replace 4 4 (cut_idx 6 (take 7 $0)))))',
         'adjust': lambda xs: 4 * proportion(xs, lambda i,o: len(i) >= 7) ,
        },
        {'concept': '(lambda (singleton (last $0)))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 1 for i,o in xs) else 0.0),
        },
        {'concept': '(lambda (droplast 1 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 1 for i,o in xs) else 0.0),
        },
        {'concept': '(lambda (drop (first $0) (drop 1 $0)))',
         'adjust': lambda xs: (3.0 if all(i[0] <= len(i)-1 for i,o in xs) else 0.0),
        },
        {'concept': '(lambda (drop 1 (droplast 1 $0)))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 2 for i,o in xs) else 0.0),
        },
        {'concept': '(lambda (cons 9 (append $0 7)))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda (append (drop 1 $0) (first $0)))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 1 for i,o in xs) else 0.0),
        },
        {'concept': '(lambda (cons (last $0) (append (drop 1 (droplast 1 $0)) (first $0))))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 2 for i,o in xs) else 0.0),
        },
        {'concept': '(lambda (concat $0 (cons 7 (cons 3 (cons 8 (cons 4 (singleton 3)))))))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda (concat (cons 9 (cons 3 (cons 4 (singleton 0)))) (concat $0 (cons 7 (cons 2 (cons 9 (singleton 1)))))))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda (concat $0 $0))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda (map (lambda (+ 2 $0)) $0))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda (flatten (map (lambda (cons $0 (singleton $0))) $0)))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda (mapi + $0))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda (filter (lambda (> $0 7)) $0))',
         "adjust": lambda xs: min(1.0, 1.0/(len(xs)-2)*sum((len(o)/len(i) <= 0.75 if len(i) > 0 else 1) for i, o in xs)),
        },
        {'concept': '(lambda (filteri (lambda (lambda (is_odd $1))) $0))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda (cons (max $0) (cons (last $0) (cons (length $0) (cons (first $0) (singleton (min $0)))))))',
         "adjust": lambda xs: 2 * proportion_set(xs, lambda i,o: len(i)) ,
        },
        {'concept': '(lambda (singleton (length $0)))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda (singleton (max $0)))',
         "adjust": lambda xs: 4 * proportion_set(xs, lambda i,o: o[0]) ,
        },
        {'concept': '(lambda (singleton (sum $0)))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda (reverse $0))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda (singleton (third $0)))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 3 for i, o in xs) else 0)
        },
        {'concept': '(lambda (if (> 3 (length $0)) empty (singleton (third $0))))',
         'adjust': lambda xs: (3.0 if 0.6 >= sum(len(i) >= 3 for i, o in xs)/len(xs) >= 0.4 else 0),
        },
        {'concept': '(lambda (singleton (nth 7 $0)))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 7 for i, o in xs) else 0)
        },
        {'concept': '(lambda (if (> 7 (length $0)) empty (singleton (nth 7 $0))))',
         'adjust': lambda xs: (3.0 if 0.6 >= sum(len(i) >= 7 for i, o in xs)/len(xs) >= 0.4 else 0),
        },
        {'concept': '(lambda (singleton (nth (first $0) (drop 1 $0))))',
         'adjust': lambda xs: (3.0 if all(i[0] <= len(i)-1 for i, o in xs) else 0),
        },
        {'concept': '(lambda (swap 1 4 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 4 for i, o in xs) else 0),
        },
        {'concept': '(lambda (swap 2 3 $0))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 3 for i, o in xs) else 0),
        },
        {'concept': '(lambda (if (== (second $0) (third $0)) (swap 1 4 $0) (swap 2 3 $0)))',
         'adjust': lambda xs: 4 * proportion(xs, lambda i,o: len(i) >= 4 and ((i[1] == i[2] and i[0] != i[3]) or (i[1] != i[2] and i[0] == i[3]))) + 2 / center(xs, lambda i,o: i[1] == i[2])
        },
        {'concept': '(lambda (if (> (second $0) (third $0)) (swap 2 3 $0) (swap 1 4 $0)))',
         'adjust': lambda xs: 4 * proportion(xs, lambda i,o: len(i) >= 4 and ((i[1] > i[2] and i[0] <= i[3]) or (i[1] <= i[2] and i[0] > i[3]))) + 2 / center(xs, lambda i,o: i[1] > i[2]),
        },
        {'concept': '(lambda (cons 18 (cons 42 (cons 77 (cons 20 (singleton 36))))))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda (cons 81 (cons 99 (cons 41 (cons 23 (cons 22 (cons 75 (cons 68 (cons 30 (cons 24 (singleton 69)))))))))))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda (cons 92 (cons 63 (cons 34 (cons 18 (cons 55 $0))))))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda (repeat (first $0) 10))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda (concat (slice 3 4 $0) (concat (take 2 $0) (drop 4 $0))))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 4 for i,o in xs) else 0)
        },
        {'concept': '(lambda (drop 1 (droplast 1 $0)))',
         'adjust': lambda xs: (3.0 if all(len(i) >= 2 for i,o in xs) else 0)
        },
        {'concept': '(lambda (cons 98 (append $0 37)))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda (concat (cons 11 (cons 21 (cons 43 (singleton 19)))) (concat $0 (cons 7 (cons 89 (cons 0 (singleton 57)))))))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda (mapi + $0))',
         'adjust': lambda xs: 1.0,
        },
        {'concept': '(lambda (filter (lambda (> $0 49)) $0))',
         "adjust": lambda xs: min(1.0, 1.0/(len(xs)-2)*sum((len(o)/len(i) <= 0.75 if len(i) > 0 else 1) for i, o in xs)),
        },
        {'concept': '(lambda (reverse $0))',
         'adjust': lambda xs: 1.0,
        },
    ]

def sample_examples_greedy_parallel(p,adjust,l_min,l_max,n=10,n_restarts=1000,n_tries=1000,small=False, limit_xs_size=False):
    bests = Parallel(n_jobs = -1, verbose = 10)(delayed(greedy_set)(p, adjust, n, n_tries, small, l_min, l_max, limit_xs_size) for _ in range(n_restarts))
    return max(bests, key = lambda x: x[1])[0]

def sample_examples_greedy(p,adjust,l_min,l_max,n=10,n_restarts=10000,n_tries=100,small=False, limit_xs_size=False):
    best_score = 0.0
    best_s = None
    for i_restart in range(n_restarts):
        s, score = greedy_set(p,adjust,n,n_tries,small,l_min,l_max,limit_xs_size)
        if score > best_score:
            best_s = s
            best_score = score
            print(f"{i_restart}. {best_score}")
        else: 
            print(f"{i_restart}. no change")
    return best_s

def greedy_set(p,adjust,n,n_tries,small,l_min,l_max,limit_xs_size):
    s = initialize_set(p,n,small,l_min,l_max,limit_xs_size)
    score = score_set(s, adjust)
    for i_try in range(n_tries):
        i = sample_input(small, l_min=l_min, l_max=l_max)
        if i not in list(zip(*s))[0]:
            try:
                o = p.runWithArguments([i])
            except:
                continue
            if valid_output(o, small, limit_xs_size=limit_xs_size):
                idx = random.randint(0, n-1)
                new_s = s[:]
                new_s[idx] = (i,o)
                new_score = score_set(new_s, adjust)
                if new_score > score:
                    s = new_s
                    score = new_score
    return s, score

def initialize_set(p,n,small,l_min,l_max,limit_xs_size):
    s = []
    while len(s) < n:
        i = sample_input(small, l_min=l_min, l_max=l_max)
        try:
            o = p.runWithArguments([i])
        except:
            continue
        if valid_output(o, small, limit_xs_size=limit_xs_size) and (len(s) == 0 or i not in list(zip(*s))[0]):
            s.append((i,o))
    return s

def sample_examples_parallel(p,adjust,n=10,n_pools=1000,n_tries=20,n_sets=1000,small=False):
    def helper2(pool):
        s = make_example_set(pool, n)
        score = score_set(s, adjust)
        return score, s
    def helper1():
        best_score = 0.0
        best_s = None
        pool = build_pool(p, n_tries, False, small)
        return max((helper2(pool) for _ in range(n_sets)), key=lambda x: x[0])
    bests = Parallel(n_jobs=-1)(delayed(helper1)() for _ in range(n_pools))
    return max(bests, key=lambda x: x[0])[1]

def sample_examples(p,adjust,n=10,n_pools=1000,n_tries=20,n_sets=1000,verbose=True,small=False):
    best_score = 0.0
    best_s = None
    for i_pool in range(n_pools):
        if verbose:
            print(f"{i_pool}. ", end="")
        pool = build_pool(p, n_tries, verbose, small)
        for scanned in range(n_sets):
            s = make_example_set(pool, n)
            score = score_set(s, adjust)
            if score > best_score:
                if verbose:
                    print(f"    {scanned}: {score}")
                best_score = score
                best_s = s
    return best_s

def build_pool(p, n_tries, verbose, small):
    if verbose:
        print("building pool", end="", flush=True)
    try:
        pool = [[[([], p.runWithArguments([[]]))]]]
    except(IndexError, ValueError):
        pool = [[]]
    for length in range(1,11):
        if verbose:
            print(".", end="", flush=True)
        subpool = []
        for repetitions in range(length):
            subsubpool = []
            os = []
            tries = 0
            while len(subsubpool) < n_tries and tries < 100:
                tries += 1
                i = sample_input(small, length, repetitions)
                try:
                    o = p.runWithArguments([i])
                    if valid_output(o, small) and (i, o) not in subsubpool and os.count(o) < n_tries/10:
                        tries = 0
                        os.append(o)
                        subsubpool.append((i,o))
                except(IndexError, ValueError):
                    continue
            subpool.append(subsubpool)
        pool.append(subpool)
    if verbose:
        print("done")
    return pool

def make_example_set(pool, n):
    def helper():
        examples = []
        ls = set()
        outputs = []
        while len(examples) < n:
            if len(ls) == len(pool):
                return
            length = random.randint(0, len(pool)-1)
            ls.add(length)
            if len(pool[length]) == 0:
                continue
            subpool = random.choice(pool[length])
            for i, o in subpool:
                if (i,o) not in examples and o not in outputs:
                    examples.append((i,o))
                    outputs.append(o)
                    ls = set()
                    break
            if len(ls) == 0:
                continue
            grouped_outputs = dict((lambda xs: (tuple(xs[0]), len(list(xs[1]))))(xs)
                                for xs in itertools.groupby(sorted(outputs)))
            least_common = [list(k) for k,v in grouped_outputs.items()
                            if v == min(grouped_outputs.values())]
            for i, o in subpool:
                if o in least_common and (i, o) not in examples:
                    examples.append((i,o))
                    outputs.append(o)
                    ls = set()
                    break
        return examples
    examples = None
    while not examples:
        examples = helper()
    return examples

def valid_output(xs, small, limit_xs_size=True):
    valid_xs_size = len(xs) <= 15 if limit_xs_size else True
    return len(xs) == 0 or (valid_xs_size and max(xs) < (10 if small else 100) and min(xs) >= 0)

def score_set(s, adjust):
    (inputs, outputs) = zip(*s)
    n = len(s)

    # Measure the distribution of output lengths
    out_ws = [sum(len(o) == l for o in outputs) for l in range(11)]
    foil = [len(s)//11 + (1 if x < len(s) % 11 else 0) for x in range(11)]
    out_len = simple_entropy(out_ws)/simple_entropy(foil)

    # Inputs are unique by construction.
    # Measure the proportion of unique outputs
    unique = len(list(itertools.groupby(outputs)))/n

    # Measure the proportion of non-trivial i/o pairs
    nontrivial = sum(i != o for i,o in s)/n

    # Measure the distribution of list elements.
    all_items = _flatten(_flatten(s))
    ws = [sum(i == j for i in all_items) for j in range(100)]
    foil = [len(all_items)//100 + (1 if x < len(all_items) % 100 else 0) for x in range(100)]
    span = simple_entropy(ws)/simple_entropy(foil)

    # Measure the distribution over input lengths & repetitions
    # lrs = [(len(i), len(i)-len(set(i))) for i in inputs]
    # lr_ws = [len(list(x)) for x in itertools.groupby(sorted(lrs))]
    # foil = [len(lrs)//46 + (1 if x < len(lrs) % 46 else 0) for x in range(46)]
    # combos = simple_entropy(lr_ws)/simple_entropy(foil)

    # Measure the distribution over input lengths
    in_ws = [sum(len(i) == l for i in inputs) for l in range(11)]
    foil = [len(s)//11 + (1 if x < len(s) % 11 else 0) for x in range(11)]
    in_len = simple_entropy(in_ws)/simple_entropy(foil)

    # Adjust the score if necessary.
    adjustment = 0 if adjust is None else adjust(s)

    # print(f"{out_len/5} {unique/5} {nontrivial/5} {span/5} {in_len/5} {adjustment}")
    return (out_len + unique + nontrivial + span + in_len)/5 + adjustment

def order_examples(xs, n_orders, n_tries):
    orders = []
    for _ in range(max(n_orders, n_tries)):
        candidate = random.sample(xs, len(xs))
        orders.append((score_order(candidate), candidate))
    ranked = sorted(orders, key= lambda x: x[0])
    best = []
    while len(best) < n_orders:
        try:
            s, candidate = ranked.pop()
        except IndexError:
            break
        firsts = [order[0] for order in best]
        start = [{tuple(i) for i,o in order[:5]} for order in best]
        cand_set = {tuple(i) for i,o in candidate[:5]}
        if (candidate not in best and
            candidate[0] not in firsts and
            (len(start) == 0 or
             max(len(cand_set.intersection(s)) for s in start) <= 2)):
            best.append(candidate)
    return best

def score_order(xs):
    first_short = 1 - (abs(5 - len(xs[0][0])) / 6)
    first_informative = 1 if xs[0][0] != xs[0][1] else 0
    good_start = score_set(xs[:5], adjust=lambda xs: 0.0 )/5
    good_finish = score_set(xs[5:], adjust=lambda xs: 0.0 )/5
    return 2 * first_short + first_informative + 2 * good_start + good_finish

def flip(p=0.5):
    return random.random() < p

def sample_element(small):
    if small or flip(0.5):
        return random.randint(0, 9)
    return random.randint(0, 99)

def sample_input(small, l=None, r=None, l_min=0, l_max=10):


    length = random.randint(l_min, l_max) if l is None else l

    if small:
        repetitions = (random.randint(max(0, length-10), length-1) if r is None else r) if length > 1 else 0
        xs = set()
        while len(xs) < min(length-repetitions, 10):
            # print("len(xs)", len(xs))
            # print("length-repetitions", length-repetitions)
            # print("xs", xs)
            xs.add(sample_element(small))
    else:
        repetitions = (random.randint(max(0, length-100), length-1) if r is None else r) if length > 1 else 0
        xs = set()
        while len(xs) < min(length-repetitions, 100):
            xs.add(sample_element(small))

    xs = list(xs)
    xs.extend([random.choice(xs) for _ in range(repetitions)])
    random.shuffle(xs)
    # print('xs returned')
    return xs

def simple_entropy(ws):
    z = sum(ws)
    return -sum(w/z*math.log2(w/z) for w in ws if w > 0)

def list_primitives():
    print("Primitives:")
    for primitive in Primitive.GLOBALS:
        print(f"- {primitive}")

def sample_programs(g, type_of_sample, n=10):
    return [grammar.sample(type_of_sample, maximumDepth=10) for _ in range(n)]

def test_p_with_i(e, i):
    #print(f"e = {e}")
    p = Program.parse(e)
    #print(f"p = {p}")
    #print(f"i = {i}")
    o = p.runWithArguments([i])
    #print(f"o = {o}")
    return o

def process(dirname, i, c, n_trials=10, n_orders=2, verbose=True, small=False, human=False, kind="greedy", l_min=0, l_max=10, limit_xs_size=True):
    Primitive.GLOBALS.clear()
    grammar = Grammar.uniform(primitives())
    tp = arrow(tlist(tint), tlist(tint))
    p = Program.parse(c['concept'])
    if verbose:
        print(f"{i}. [`{p}`](./json/c{i:03}.json)", flush=True)
    if not p.canHaveType(tp):
        if verbose:
            print(f"    incorrect type {p.infer()}", flush=True)
        return
    if human:
        examples = [(inp, p.runWithArguments([inp])) for inp in c['inputs']]
    elif kind == "parallel":
        examples = sample_examples_parallel(p, c["adjust"], n=n_trials, n_pools=1000, n_tries=20, n_sets=1000, small=small)
    elif kind == "greedy":
        examples = sample_examples_greedy(p, c["adjust"], l_min=l_min, l_max=l_max, n=n_trials, n_restarts=20, n_tries=100, small=small, limit_xs_size=limit_xs_size)
    else:
        examples = sample_examples(p, c["adjust"], n=n_trials, n_pools=1000, n_tries=20, n_sets=1000, verbose=verbose, small=small)
    if n_orders == 0:
        data = {
            'id': f"c{i:03}",
            'program': c['concept'],
            'data': [{"i": i, "o": o} for i,o in examples]
        }
        with open(f"{dirname}/c{i:03}_1.json", "w") as fd:
            fd.write(json.dumps(data))
    else:
        for i_order, order in enumerate(order_examples(examples, n_orders, 5000), 1):
            data = {
                'id': f"c{i:03}",
                'program': c['concept'],
                'data': [{"i": i, "o": o} for i,o in order]
                }
            with open(f"{dirname}/c{i:03}_{i_order}.json", "w") as fd:
                fd.write(json.dumps(data))

def process_2(programs, n_trials=1000, small=False):
    Primitive.GLOBALS.clear()
    grammar = Grammar.uniform(primitives())
    inputs = []
    while len(inputs) < n_trials:
        i = sample_input(small)
        if i not in inputs:
          inputs.append(i)
    pairss = {}
    for program in programs:
        p = Program.parse(program)
        s = ""
        for i in inputs:
            try:
                s += f" {str((i, p.runWithArguments([i])))} "
            except:
                s += f" ({i}, ERR) "
        if s not in pairss:
            pairss[s] = [p]
        else:
            pairss[s].append(p)
    return pairss

def count_applications(program):
    return sum(subprogram[1].isApplication for subprogram in program.walk())

def ilogit(x):
    return 1/(1+math.exp(-x))

def predict(program, visible, semi):
    Primitive.GLOBALS.clear()
    grammar = Grammar.uniform(primitives())
    request =  arrow(tlist(tint), tlist(tint))
    p = Program.parse(program)
    apps = count_applications(p)
    length = p.size()
    depth = p.depth()
    print(f"{program},{length},{depth},1,{apps},{visible},{semi}")


def list_priors(filename, programs, small=False):
    Primitive.GLOBALS.clear()
    grammar = Grammar.uniform(primitives())
    request =  arrow(tlist(tint), tlist(tint))
    with open(filename, 'w') as fd:
        fd.write("id,program,prior,length,depth,lambda,apps,visible,semi,hidden\n")
        for i, program in enumerate(programs):
            p = Program.parse(program["concept"])
            try:
                prior = grammar.logLikelihood(request, p)
            except AssertionError:
                prior = "NA"
            apps = count_applications(p)
            fd.write(f"c{(i+1):03},{program['concept']},{prior},{p.size()},{p.depth()},1,{apps},,\n")

def make_grammar():
    Primitive.GLOBALS.clear()
    return Grammar.uniform(primitives())

def robustfill_fill(data):
    programs = []
    cpus = []
    counts = []
    for x in data.iterrows():
        if x[1]['program'] == np.nan:
            programs.append("(lambda $0)")
            cpus.append(0)
            counts.append(0)
        else:
            programs.append(x[1]['program'])
            cpus.append(x[1]['cpu'])
            counts.append(x[1]['count'])
    data['program'] = programs
    data['cpu'] = cpus
    data['count'] = counts
    return data

def compute_output_and_accuracy(data):
    Primitive.GLOBALS.clear()
    grammar = Grammar.uniform(model_comparison_primitives_99())
    accuracys = []
    responses = []
    for x in data.iterrows():
        try:
            response = test_p_with_i(x[1]['program'],literal_eval(x[1]['input']))
        except IndexError:
            response = None
        except RecursionDepthExceeded:
            response = None
        except AttributeError:
            reponse = None
        responses.append(response)
        accuracys.append(response == literal_eval(x[1]['output']))
    data['response'] = responses
    data['accuracy'] = accuracys
    return data

def wave_3_1_ids():
    return ["c002", "c005", "c015", "c027", "c028", "c030", "c031", "c032", "c035", "c036", "c039", "c040", "c060", "c076", "c078", "c088", "c089"]

def adjust_id(id, purpose):
    if purpose == "model":
        return id
    else:
        i = int(id[1:])+100
        return f"c{i:03}"

def pretty_print(e):
    Primitive.GLOBALS.clear()
    grammar = Grammar.uniform(primitives())
    return(prettyProgram(Program.parse(e), Lisp=True))


def print_table():
    filename = "~/sync/josh/library/phd/thesis/analyses/table.csv"
    stimuli = "~/sync/josh/library/phd/thesis/analyses/stimuli.csv"
    df1 = pd.read_csv(filename)
    df2 = pd.read_csv(stimuli)
    r = re.compile('𝜆')
    with open("/Users/rule/table.txt", "w") as fd:
        for x in df1.iterrows():
            fd.write(f"{x[1]['mu']} & {adjust_id(x[1]['id'], x[1]['purpose'])} & {x[1]['length']} & \\emph{{{x[1]['gloss']}}}\\\\\n")
            pretty = r.sub('@$\lambda$@', pretty_print(x[1]['program']))
            fd.write(f" & & & \\mnln{{{pretty}}}\\\\[0.5em]\n")

def process_enumeration_data():
    stimuli_filename = "~/sync/josh/library/phd/thesis/analyses/stimuli.csv"
    mc_dir = "~/sync/josh/library/research/list-routines/project/list-routine-model-comparison/waves"
    h_dir = "~/sync/josh/library/research/list-routines/project/list-routine-human-experiments/waves/1/data/enumeration"
    enumeration3_filename = mc_dir + "/3/data/enumeration/bests.csv"
    enumeration31_filename = mc_dir + "/3_1/data/enumeration/w3.1"
    enumeration31_filename = mc_dir + "/3_1/data/enumeration/w3.1"
    h_filename = h_dir + "/data.csv"
    stimuli = pd.read_csv(stimuli_filename).rename(columns={'program': 'concept'})
    enumeration_3 = (pd.read_csv(enumeration3_filename, header=None, names=['id','run','order','trial','program','cpu','count'])
                     .assign(trial=lambda x: x.trial+1,
                             purpose="model"))
    enumeration_3 = enumeration_3[enumeration_3.id.isin(wave_3_1_ids()) == False]
    enumeration_31 = (pd.read_csv(enumeration31_filename, header=None, names=['id','run','order','trial','program','cpu','count'])
                     .assign(trial=lambda x: x.trial+1,
                             purpose="model"))
    enumeration_h = (pd.read_csv(h_filename, header=None, names=['id','run','order','trial','program','cpu','count'])
                     .assign(trial=lambda x: x.trial+1,
                             purpose="dataset"))
    enumeration = pd.concat([enumeration_3, enumeration_31, enumeration_h])
    (pd.merge(stimuli, enumeration, how='left', on=['purpose','id','order','trial'])
     .reindex(columns = ["purpose", "id", "order", "run", "trial", "concept", "input", "output", "program", "cpu", "count"])
     .pipe(compute_output_and_accuracy)
     .to_csv("enumeration_data.csv", header=True, index=False))

def process_fleet_data():
    Primitive.GLOBALS.clear()
    grammar = Grammar.uniform(model_comparison_primitives_99())
    stimuli_filename = "~/sync/josh/library/phd/thesis/analyses/stimuli.csv"
    fleet_dirname1 = "/Users/rule/sync/josh/library/research/list-routines/project/list-routine-model-comparison/waves/3_1/data/fleet/out"
    fleet_dirname2 = "/Users/rule/sync/josh/library/research/list-routines/project/list-routine-human-experiments/waves/1/data/fleet/out"
    stimuli = pd.read_csv(stimuli_filename).rename(columns={'program': 'concept'})
    fleets = []
    for filename in glob.iglob(fleet_dirname1 + "/*predictions.txt"):
        tmp = (pd.read_csv(filename, delimiter="\t", header=None, names=['file','run','trial','cpu','count','program'])
                 .assign(trial=lambda x: x.trial + 1,
                         run=lambda x: x.run + 1,
                         program=lambda x: "(lambda (fix $0 (lambda " + x.program.str.replace("fix", "$1") + ")))",
                         id=lambda x: x.file.str.extract(r'\/(c\d{3})_'),
                         purpose="model",
                         order=lambda x: pd.to_numeric(x.file.str.extract(r'\/c\d{3}_(\d)', expand = False)))
                 .drop(['file'], axis=1))
        fleets.append(tmp)
    for filename in glob.iglob(fleet_dirname2 + "/*predictions.txt"):
        tmp = (pd.read_csv(filename, delimiter="\t", header=None, names=['file','run','trial','cpu','count','program'])
                 .assign(trial=lambda x: x.trial + 1,
                         run=lambda x: x.run + 1,
                         program=lambda x: "(lambda (fix $0 (lambda " + x.program.str.replace("fix", "$1") + ")))",
                         id=lambda x: x.file.str.extract(r'\/(c\d{3})_'),
                         purpose="dataset",
                         order=lambda x: pd.to_numeric(x.file.str.extract(r'\/c\d{3}_(\d)', expand = False)))
                 .drop(['file'], axis=1))
        fleets.append(tmp)
    fleet = pd.concat(fleets)
    (pd.merge(stimuli, fleet, how='left', on=['purpose','id','order','trial'])
     .pipe(compute_output_and_accuracy)
     .to_csv("fleet_data.csv", header=True, index=False))

def process_metagol_data():
    Primitive.GLOBALS.clear()
    grammar = Grammar.uniform(model_comparison_primitives_99())
    results3_filename = "/Users/rule/sync/josh/library/research/list-routines/project/list-routine-model-comparison/waves/3_1/data/metagol/cropper-results/results3.csv"
    preds3_filename = "/Users/rule/sync/josh/library/research/list-routines/project/list-routine-model-comparison/waves/3/data/metagol/predictions.csv"
    results31_filename = "/Users/rule/sync/josh/library/research/list-routines/project/list-routine-model-comparison/waves/3_1/data/metagol/cropper-results/results3-1.csv"
    results312_filename = "/Users/rule/sync/josh/library/research/list-routines/project/list-routine-model-comparison/waves/3_1/data/metagol/cropper-results/results3-1-c005.csv"
    preds31_filename = "/Users/rule/sync/josh/library/research/list-routines/project/list-routine-model-comparison/waves/3_1/data/metagol/predictions.csv"
    stimuli_filename = "~/sync/josh/library/phd/thesis/analyses/stimuli.csv"
    metagol_3 = (pd.read_csv(results3_filename,
                           delimiter=", ",
                           header=None,
                           engine='python',
                           names=['task','run','order','trial','input','output','response','accuracy'])
               .assign(id=lambda x: x.task.apply(str).str.replace('(\d+)', lambda m: f"c{int(m.group(0)):03}"),
                       accuracy=lambda x: 1 * x.accuracy))
    metagol_3 = metagol_3[metagol_3.id.isin(wave_3_1_ids()) == False]
    metagol_31 = (pd.read_csv(results31_filename,
                           delimiter=", ",
                           header=None,
                           engine='python',
                           names=['task','run','order','trial','input','output','response','accuracy'])
               .assign(id=lambda x: x.task.apply(str).str.replace('(\d+)', lambda m: f"c{int(m.group(0)):03}"),
                       run=lambda x: x.run - 1,
                       accuracy=lambda x: 1 * x.accuracy))
    metagol_312 = (pd.read_csv(results312_filename,
                           delimiter=", ",
                           header=None,
                           engine='python',
                           names=['task','run','order','trial','input','output','response','accuracy'])
                  .assign(id=lambda x: x.task.apply(str).str.replace('(\d+)', lambda m: f"c{int(m.group(0)):03}"),
                          run=lambda x: x.run - 1,
                          accuracy=lambda x: 1 * x.accuracy))
    metagol = pd.concat([metagol_3, metagol_31, metagol_312])
    counts_3 = (pd.read_csv(preds3_filename,
                          header=None,
                          names=['task','run','order','trial','program','cpu','count'])
                .assign(id=lambda x: x.task.apply(str).str.replace('(\d+)', lambda m: f"c{int(m.group(0)):03}"))
                .drop(['program','count'], axis=1))
    counts_3 = counts_3[counts_3.id.isin(wave_3_1_ids()) == False]
    counts_31 = (pd.read_csv(preds31_filename,
                          header=None,
                          names=['task','run','order','trial','program','cpu','count'])
                 .assign(id=lambda x: x.task.apply(str).str.replace('(\d+)', lambda m: f"c{int(m.group(0)):03}"))
                 .drop(['program','count'], axis=1))
    counts = pd.concat([counts_3, counts_31])
    stimuli = (pd.read_csv(stimuli_filename)
               .query('purpose == "model"')
               .drop(['purpose'], axis=1)
               .rename(columns={'program': 'concept'}))
    metagol = (pd.merge(metagol, counts, how='left', on=['id','task','trial','order','run'])
               .assign(input = lambda x: x.input.str.replace(',', ', '),
                       response = lambda x: x.response.str.replace(',', ', '),
                       output = lambda x: x.output.str.replace(',', ', ')))
    (pd.merge(stimuli, metagol, how='left', on=['id','order','trial','input','output'])
     .drop(['task'], axis=1)
     .to_csv("metagol_data.csv", header=True, index=False))

def process_robustfill_data():
    Primitive.GLOBALS.clear()
    grammar = Grammar.uniform(model_comparison_primitives_99())

    stimuli_filename = "~/sync/josh/library/phd/thesis/analyses/stimuli.csv"
    stimuli = (pd.read_csv(stimuli_filename)
               .rename(columns={'program': 'concept'})
               .query('purpose == "model"')
               .drop(['purpose'], axis=1))

    rf_dirname = "~/sync/josh/library/research/list-routines/project/list-routine-model-comparison/waves/3_1/data/robustfill"
    rf3_filename = rf_dirname + "/predictions_wave_3_complete.csv"
    rf_3 = pd.read_csv(rf3_filename)
    rf_3 = rf_3[rf_3.id.isin(wave_3_1_ids()) == False]
    rf31_filename = rf_dirname + "/predictions_wave_31_complete.csv"
    rf_31 = pd.read_csv(rf31_filename).query('id != "c076"')
    rf76_filename = rf_dirname + "/predictions_wave_31_76_complete.csv"
    rf_76 = pd.read_csv(rf76_filename)
    rf = pd.concat([rf_3, rf_31, rf_76])
    (pd.merge(stimuli, rf, how='left', on=['id','order','trial'])
     .sort_values(['id','order','run','trial'], axis=0)
     .pipe(compute_output_and_accuracy)
     .to_csv("robustfill_data.csv", header=True, index=False))

def process_hl_data():
    Primitive.GLOBALS.clear()
    grammar = Grammar.uniform(model_comparison_primitives_99())
    stimuli_filename = "/Users/rule/sync/josh/projects/research/text/learning_list_functions/data/stimuli.csv"
    hl_dirname = "/Users/rule/sync/josh/projects/research/data/2020-12-10-15-45-09/out"
    stimuli = (pd.read_csv(stimuli_filename)
               .rename(columns={'program': 'concept'}))
    hls = []
    for filename in glob.iglob(hl_dirname + "/model/*predictions.csv"):
        tmp = (pd.read_csv(filename, delimiter=",", header=0)
                 .assign(run=lambda x: x.run + 1,
                         purpose= lambda x: 'model',
                         id=lambda x: x.problem)
                 .drop(['problem'], axis=1))
        hls.append(tmp)
    for filename in glob.iglob(hl_dirname + "/dataset/*predictions.csv"):
        tmp = (pd.read_csv(filename, delimiter=",", header=0)
                 .assign(run=lambda x: x.run + 1,
                         purpose= lambda x: 'dataset',
                         id=lambda x: x.problem)
                 .drop(['problem'], axis=1))
        hls.append(tmp)
    hl = pd.concat(hls)
    (pd.merge(hl, stimuli, how='left', on=['id','purpose','order','trial'])
     .to_csv("hl_2020-12-10-15-45-09.csv", header=True, index=False))



if __name__ == "__main__":

    ## Generating new data for BIG-bench

    for i, c in enumerate(model_comparison_wave_3_long(), 1):
        process("/tmp",
                i,
                c,
                n_trials=10,
                n_orders=0,
                verbose=True,
                small=(i <= 80 and i != 76 and i != 77 and i != 73),
                human=False,
                kind="greedy",
                l_min=50,
                l_max=100,
                limit_xs_size=False
        )

    # for i, c in enumerate(human_experiments_wave_1(), 101):
    #     process("./tmp",
    #             i,
    #             c,
    #             n_trials=32,
    #             n_orders=0,
    #             verbose=True,
    #             small=False,
    #             human=False,
    #             kind="greedy",
    #     )

    ## Human Experiment - Wave Pilot
    #
    # for i, c in enumerate(wave_pilot(), 1):
    #     process("../waves/pilot/json/human", i, c, n_trials=11, n_orders=2, verbose=True, small=False, human=True)
    #     process("../waves/pilot/json/machine", i, c, n_trials=11, n_orders=2, verbose=True, small=False, human=False, parallel=True)

    ## Human Experiment - Wave 1

    # for i, c in enumerate(human_experiments_wave_1(), 1):
    #     process("./tmp_dataset",
    #             i,
    #             c,
    #             n_trials=64,
    #             n_orders=1,
    #             verbose=True,
    #             small=False,
    #             human=False,
    #             kind="greedy",
    #             )

    # list_priors("dataset_priors.csv", human_experiments_wave_1())

    ## Model Comparison - Wave 3

    # for i, c in enumerate(model_comparison_wave_3(), 1):
    #     if i in [2,5,15,27,28,30,31,32,35,36,39,40,60,76,78,88,89]:
    #         process("./tmp",
    #                 i,
    #                 c,
    #                 n_trials=11,
    #                 n_orders=5,
    #                 verbose=True,
    #                 small=(i <= 80),
    #                 human=False,
    #                 kind="greedy",
    #         )

    # list_priors("model_comparison_priors.csv", model_comparison_wave_3())

    ## Uniqueness checks for all thesis concepts

    # mc_ps = [x['concept'] for x in model_comparison_wave_3()]
    # hd_ps = [x['concept'] for x in human_experiments_wave_1()]

    # ps_0_09 = mc_ps[0:80]
    # ps_0_99 = mc_ps[80:] + hd_ps

    # blah = process_2(ps_0_09, small=True)
    # print(len(blah))
    # for v in blah.values():
    #     print(f"{len(v)} => {v}")

    # print("\n\n\n")

    # blah = process_2(ps_0_99, small=False)
    # print(len(blah))
    # for v in blah.values():
    #     print(f"{len(v)} => {v}")

