from dreamcoder.program import Primitive, Program
from dreamcoder.grammar import Grammar
from dreamcoder.type import tlist, tint, tbool, arrow, t0, t1, t2

from collections import OrderedDict
import math
from functools import reduce

def _plusNatural(x):
    def g(y):
        if y < 0 or x < 0: raise ValueError()
        return x + y
    return g
def _minusNatural(x):
    def g(y):
        if y < 0 or x < 0: raise ValueError()
        return x - y
    return g

def _plusNatural9(x):
    def g(y):
        if y < 0 or x < 0 or y > 9 or x > 9: raise ValueError()
        return x + y
    return g
def _minusNatural9(x):
    def g(y):
        if g < 0 or x < 0 or y > 9 or x > 9: raise ValueError()
        return x - y
    return g

def _plusNatural99(x):
    def g(y):
        if y < 0 or x < 0 or y > 99 or x > 99: raise ValueError()
        return x + y
    return g
def _minusNatural99(x):
    def g(y):
        if g < 0 or x < 0 or y > 99 or x > 99: raise ValueError()
        return x - y
    return g
        

def _flatten(l): return [x for xs in l for x in xs]

def _range(n):
    if n < 100: return list(range(n))
    raise ValueError()
def _if(c): return lambda t: lambda f: t if c else f


def _and(x): return lambda y: x and y


def _or(x): return lambda y: x or y


def _addition(x): return lambda y: x + y


def _subtraction(x): return lambda y: x - y


def _multiplication(x): return lambda y: x * y


def _division(x): 
    def g(y):
        if y == 0:
            raise ValueError
        return x // y
    return g


def _negate(x): return -x


def _reverse(x): return list(reversed(x))


def _append(x): return lambda y: x + y


def _cons(x): return lambda y: [x] + y


def _car(x): return x[0]


def _cdr(x): return x[1:]


def _isEmpty(x): return x == []


def _single(x): return [x]


def _slice(x): return lambda y: lambda l: l[x:y]


def _map(f): return lambda l: list(map(f, l))


def _zip(a): return lambda b: lambda f: list(map(lambda x,y: f(x)(y), a, b))


def _mapi(f): return lambda l: list(map(lambda i_x: f(i_x[0])(i_x[1]), enumerate(l)))


def _reduce(f): return lambda x0: lambda l: reduce(lambda a, x: f(a)(x), l, x0)


def _reducei(f): return lambda x0: lambda l: reduce(
    lambda a, t: f(t[0])(a)(t[1]), enumerate(l), x0)


def _fold(l): return lambda x0: lambda f: reduce(
    lambda a, x: f(x)(a), l[::-1], x0)


def _foldi(l): return lambda x0: lambda f: reduce(
    lambda a, t: f(t[0])(a)(t[1]), enumerate(l[::-1]), x0)


def _eq(x): return lambda y: x == y


def _eq0(x): return x == 0


def _a1(x): return x + 1


def _d1(x): return x - 1


def _mod(x): return lambda y: x % y


def _isEven(x): return x % 2 == 0


def _isOdd(x): return x % 2 == 1


def _not(x): return not x


def _gt(x): return lambda y: x > y


def _lt(x): return lambda y: x < y


def _index(j): return lambda l: l[j]


def _replace(f): return lambda lnew: lambda lin: _flatten(
    lnew if f(i)(x) else [x] for i, x in enumerate(lin))


def _isPrime(n):
    return n in {
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
        101,
        103,
        107,
        109,
        113,
        127,
        131,
        137,
        139,
        149,
        151,
        157,
        163,
        167,
        173,
        179,
        181,
        191,
        193,
        197,
        199}


def _isSquare(n):
    return int(math.sqrt(n)) ** 2 == n


def _appendmap(f): lambda xs: [y for x in xs for y in f(x)]


def _filter(f): return lambda l: list(filter(f, l))


def _filteri(f):
    def g(l):
        ans = []
        for i,x in enumerate(l):
            if not f(i)(x):
                ans.append(x)
        return ans
    return g



def _any(f): return lambda l: any(f(x) for x in l)


def _all(f): return lambda l: all(f(x) for x in l)


def _find(x):
    def _inner(l):
        try:
            return l.index(x)
        except ValueError:
            return -1
    return _inner


def _unfold(x): return lambda p: lambda h: lambda n: __unfold(p, f, n, x)


def __unfold(p, f, n, x, recursion_limit=50):
    if recursion_limit <= 0:
        raise RecursionDepthExceeded()
    if p(x):
        return []
    return [f(x)] + __unfold(p, f, n, n(x), recursion_limit - 1)


class RecursionDepthExceeded(Exception):
    pass


def _fix(argument):
    def inner(body):
        recursion_limit = [20]

        def fix(x):
            def r(z):
                recursion_limit[0] -= 1
                if recursion_limit[0] <= 0:
                    raise RecursionDepthExceeded("RecursionDepthExceeded")
                else:
                    return fix(z)

            return body(r)(x)
        return fix(argument)

    return inner


def curry(f): return lambda x: lambda y: f((x, y))


def _fix2(a1):
    return lambda a2: lambda body: \
        _fix((a1, a2))(lambda r: lambda n_l: body(curry(r))(n_l[0])(n_l[1]))


primitiveRecursion1 = Primitive("fix1",
                                arrow(t0,
                                      arrow(arrow(t0, t1), t0, t1),
                                      t1),
                                _fix)

primitiveRecursion2 = Primitive("fix2",
                                arrow(t0, t1,
                                      arrow(arrow(t0, t1, t2), t0, t1, t2),
                                      t2),
                                _fix2)


def _match(l):
    return lambda b: lambda f: b if l == [] else f(l[0])(l[1:])


def primitives():
    return [Primitive(str(j), tint, j) for j in range(6)] + [
        Primitive("empty", tlist(t0), []),
        Primitive("singleton", arrow(t0, tlist(t0)), _single),
        Primitive("range", arrow(tint, tlist(tint)), _range),
        Primitive("++", arrow(tlist(t0), tlist(t0), tlist(t0)), _append),
        # Primitive("map", arrow(arrow(t0, t1), tlist(t0), tlist(t1)), _map),
        Primitive(
            "mapi",
            arrow(
                arrow(
                    tint,
                    t0,
                    t1),
                tlist(t0),
                tlist(t1)),
            _mapi),
        # Primitive("reduce", arrow(arrow(t1, t0, t1), t1, tlist(t0), t1), _reduce),
        Primitive(
            "reducei",
            arrow(
                arrow(
                    tint,
                    t1,
                    t0,
                    t1),
                t1,
                tlist(t0),
                t1),
            _reducei),
        Primitive("true", tbool, True),
        Primitive("not", arrow(tbool, tbool), _not),
        Primitive("and", arrow(tbool, tbool, tbool), _and),
        Primitive("or", arrow(tbool, tbool, tbool), _or),
        # Primitive("if", arrow(tbool, t0, t0, t0), _if),

        Primitive("sort", arrow(tlist(tint), tlist(tint)), sorted),
        Primitive("+", arrow(tint, tint, tint), _addition),
        Primitive("*", arrow(tint, tint, tint), _multiplication),
        Primitive("negate", arrow(tint, tint), _negate),
        Primitive("mod", arrow(tint, tint, tint), _mod),
        Primitive("eq?", arrow(tint, tint, tbool), _eq),
        Primitive("gt?", arrow(tint, tint, tbool), _gt),
        Primitive("is-prime", arrow(tint, tbool), _isPrime),
        Primitive("is-square", arrow(tint, tbool), _isSquare),

        # these are achievable with above primitives, but unlikely
        # Primitive("flatten", arrow(tlist(tlist(t0)), tlist(t0)), _flatten),
        # (lambda (reduce (lambda (lambda (++ $1 $0))) empty $0))
        Primitive("sum", arrow(tlist(tint), tint), sum),
        # (lambda (lambda (reduce (lambda (lambda (+ $0 $1))) 0 $0)))
        Primitive("reverse", arrow(tlist(t0), tlist(t0)), _reverse),
        # (lambda (reduce (lambda (lambda (++ (singleton $0) $1))) empty $0))
        Primitive("all", arrow(arrow(t0, tbool), tlist(t0), tbool), _all),
        # (lambda (lambda (reduce (lambda (lambda (and $0 $1))) true (map $1 $0))))
        Primitive("any", arrow(arrow(t0, tbool), tlist(t0), tbool), _any),
        # (lambda (lambda (reduce (lambda (lambda (or $0 $1))) true (map $1 $0))))
        Primitive("index", arrow(tint, tlist(t0), t0), _index),
        # (lambda (lambda (reducei (lambda (lambda (lambda (if (eq? $1 $4) $0 0)))) 0 $0)))
        Primitive("filter", arrow(arrow(t0, tbool), tlist(t0), tlist(t0)), _filter),
        # (lambda (lambda (reduce (lambda (lambda (++ $1 (if ($3 $0) (singleton $0) empty)))) empty $0)))
        # Primitive("replace", arrow(arrow(tint, t0, tbool), tlist(t0), tlist(t0), tlist(t0)), _replace),
        # (FLATTEN (lambda (lambda (lambda (mapi (lambda (lambda (if ($4 $1 $0) $3 (singleton $1)))) $0)))))
        Primitive("slice", arrow(tint, tint, tlist(t0), tlist(t0)), _slice),
        # (lambda (lambda (lambda (reducei (lambda (lambda (lambda (++ $2 (if (and (or (gt? $1 $5) (eq? $1 $5)) (not (or (gt? $4 $1) (eq? $1 $4)))) (singleton $0) empty))))) empty $0))))
    ]


def basePrimitives():
    return [Primitive(str(j), tint, j) for j in range(6)] + [
        Primitive("*", arrow(tint, tint, tint), _multiplication),
        Primitive("gt?", arrow(tint, tint, tbool), _gt),
        Primitive("is-prime", arrow(tint, tbool), _isPrime),
        Primitive("is-square", arrow(tint, tbool), _isSquare),
        # McCarthy
        Primitive("empty", tlist(t0), []),
        Primitive("cons", arrow(t0, tlist(t0), tlist(t0)), _cons),
        Primitive("car", arrow(tlist(t0), t0), _car),
        Primitive("cdr", arrow(tlist(t0), tlist(t0)), _cdr),
        Primitive("empty?", arrow(tlist(t0), tbool), _isEmpty),
        Primitive("if", arrow(tbool, t0, t0, t0), _if),
        Primitive("eq?", arrow(tint, tint, tbool), _eq),
        Primitive("+", arrow(tint, tint, tint), _addition),
        Primitive("-", arrow(tint, tint, tint), _subtraction)
    ]

zip_primitive = Primitive("zip", arrow(tlist(t0), tlist(t1), arrow(t0, t1, t2), tlist(t2)), _zip)

def bootstrapTarget():
    """These are the primitives that we hope to learn from the bootstrapping procedure"""
    return [
        # learned primitives
        Primitive("map", arrow(arrow(t0, t1), tlist(t0), tlist(t1)), _map),
        Primitive("unfold", arrow(t0, arrow(t0,tbool), arrow(t0,t1), arrow(t0,t0), tlist(t1)), _unfold),
        Primitive("range", arrow(tint, tlist(tint)), _range),
        Primitive("index", arrow(tint, tlist(t0), t0), _index),
        Primitive("fold", arrow(tlist(t0), t1, arrow(t0, t1, t1), t1), _fold),
        Primitive("length", arrow(tlist(t0), tint), len),

        # built-ins
        Primitive("if", arrow(tbool, t0, t0, t0), _if),
        Primitive("+", arrow(tint, tint, tint), _addition),
        Primitive("-", arrow(tint, tint, tint), _subtraction),
        Primitive("empty", tlist(t0), []),
        Primitive("cons", arrow(t0, tlist(t0), tlist(t0)), _cons),
        Primitive("car", arrow(tlist(t0), t0), _car),
        Primitive("cdr", arrow(tlist(t0), tlist(t0)), _cdr),
        Primitive("empty?", arrow(tlist(t0), tbool), _isEmpty),
    ] + [Primitive(str(j), tint, j) for j in range(10)]


def bootstrapTarget_extra():
    """This is the bootstrap target plus list domain specific stuff"""
    return bootstrapTarget() + [
        Primitive("*", arrow(tint, tint, tint), _multiplication),
        Primitive("mod", arrow(tint, tint, tint), _mod),
        Primitive("gt?", arrow(tint, tint, tbool), _gt),
        Primitive("eq?", arrow(t0, t0, tbool), _eq),
        Primitive("is-prime", arrow(tint, tbool), _isPrime),
        Primitive("is-square", arrow(tint, tbool), _isSquare),
    ]

def no_length():
    """this is the primitives without length because one of the reviewers wanted this"""
    return [p for p in bootstrapTarget() if p.name != "length"] + [
        Primitive("*", arrow(tint, tint, tint), _multiplication),
        Primitive("mod", arrow(tint, tint, tint), _mod),
        Primitive("gt?", arrow(tint, tint, tbool), _gt),
        Primitive("eq?", arrow(t0, t0, tbool), _eq),
        Primitive("is-prime", arrow(tint, tbool), _isPrime),
        Primitive("is-square", arrow(tint, tbool), _isSquare),
    ]


def josh_primitives(w):
    if w == "1":
        return [
            Primitive("empty_int", tlist(tint), []),
            Primitive("cons_int", arrow(tint, tlist(tint), tlist(tint)), _cons),
            Primitive("car_int", arrow(tlist(tint), tint), _car),
            Primitive("cdr_int", arrow(tlist(tint), tlist(tint)), _cdr),
            Primitive("empty?_int", arrow(tlist(tint), tbool), _isEmpty),
            Primitive("if", arrow(tbool, t0, t0, t0), _if),
            Primitive("eq?", arrow(tint, tint, tbool), _eq),
            primitiveRecursion1
        ] + [Primitive(str(j), tint, j) for j in range(100)]
    elif w == "2":
        return [
            Primitive("empty_int", tlist(tint), []),
            Primitive("cons_int", arrow(tint, tlist(tint), tlist(tint)), _cons),
            Primitive("car_int", arrow(tlist(tint), tint), _car),
            Primitive("cdr_int", arrow(tlist(tint), tlist(tint)), _cdr),
            Primitive("empty?_int", arrow(tlist(tint), tbool), _isEmpty),
            Primitive("if", arrow(tbool, t0, t0, t0), _if),
            Primitive("eq?", arrow(tint, tint, tbool), _eq),
            Primitive("-n", arrow(tint,tint,tint), _minusNatural),
            Primitive("+n", arrow(tint,tint,tint), _plusNatural),
            Primitive("gt?", arrow(tint,tint,tbool), _gt),
            primitiveRecursion1
        ] + [Primitive(str(j), tint, j) for j in range(10)]
    elif w == "final":
        return [
            Primitive("empty", tlist(t0), []),
            Primitive("cons", arrow(t0, tlist(t0), tlist(t0)), _cons),
            Primitive("car", arrow(tlist(t0), t0), _car),
            Primitive("cdr", arrow(tlist(t0), tlist(t0)), _cdr),
            Primitive("empty?", arrow(tlist(t0), tbool), _isEmpty),
            Primitive("if", arrow(tbool, t0, t0, t0), _if),
            Primitive("eq?", arrow(tint, tint, tbool), _eq),
            Primitive("-n99", arrow(tint,tint,tint), _minusNatural99),
            Primitive("+n99", arrow(tint,tint,tint), _plusNatural99),
            Primitive("gt?", arrow(tint,tint,tbool), _gt),
            primitiveRecursion1
        ] + [Primitive(str(j), tint, j) for j in range(100)]
    elif w == "3" or w == "3.1":
        return ([
            Primitive("empty", tlist(t0), []),
            Primitive("cons", arrow(t0, tlist(t0), tlist(t0)), _cons),
            Primitive("car", arrow(tlist(t0), t0), _car),
            Primitive("cdr", arrow(tlist(t0), tlist(t0)), _cdr),
            Primitive("empty?", arrow(tlist(t0), tbool), _isEmpty),
            Primitive("if", arrow(tbool, t0, t0, t0), _if),
            # Primitive("eq?", arrow(tint, tint, tbool), _eq),
            # changed type of eq? to match eq? primitive on ocaml side
            Primitive("eq?", arrow(t0, t0, tbool), _eq),
            Primitive("-n9", arrow(tint,tint,tint), _minusNatural9),
            Primitive("+n9", arrow(tint,tint,tint), _plusNatural9),
            Primitive("gt?", arrow(tint,tint,tbool), _gt),
            primitiveRecursion1
        ] + [Primitive(str(j), tint, j) for j in range(10)],
                [
            Primitive("empty", tlist(t0), []),
            Primitive("cons", arrow(t0, tlist(t0), tlist(t0)), _cons),
            Primitive("car", arrow(tlist(t0), t0), _car),
            Primitive("cdr", arrow(tlist(t0), tlist(t0)), _cdr),
            Primitive("empty?", arrow(tlist(t0), tbool), _isEmpty),
            Primitive("if", arrow(tbool, t0, t0, t0), _if),
            Primitive("eq?", arrow(tint, tint, tbool), _eq),
            Primitive("-n99", arrow(tint,tint,tint), _minusNatural99),
            Primitive("+n99", arrow(tint,tint,tint), _plusNatural99),
            Primitive("gt?", arrow(tint,tint,tbool), _gt),
            primitiveRecursion1
        ] + [Primitive(str(j), tint, j) for j in range(100)])

    elif w == "rich_0_9":
        return [
            Primitive(str(j), tint, j) for j in range(10)
        ] + [
            Primitive("true", tbool, True),
            Primitive("false", tbool, False),
            Primitive("empty", tlist(t0), []),
            Primitive("+", arrow(tint, tint, tint), _addition),
            Primitive("-", arrow(tint, tint, tint), _subtraction),
            Primitive("*", arrow(tint, tint, tint), _multiplication),
            Primitive("/", arrow(tint, tint, tint), _division),
            Primitive("mod", arrow(tint, tint, tint), _mod),
            Primitive("gt?", arrow(tint, tint, tbool), _gt),
            Primitive("lt?", arrow(tint, tint, tbool), _lt),
            Primitive("is-even", arrow(tint, tbool), _isEven),
            Primitive("is-odd", arrow(tint, tbool), _isOdd),
            Primitive("and", arrow(tbool, tbool, tbool), _and),
            Primitive("or", arrow(tbool, tbool, tbool), _or),
            Primitive("not", arrow(tbool, tbool), _not),
            Primitive("if", arrow(tbool, t0, t0, t0), _if),
            Primitive("eq?", arrow(t0, t0, tbool), _eq),
            Primitive("singleton", arrow(t0, tlist(t0)), _single),
            Primitive("repeat", arrow(t0, tint, tlist(t0)), _repeat),
            Primitive("range", arrow(tint, tint, tint, tlist(tint)), _rangeGeneral),
            Primitive("append", arrow(tlist(t0), t0, tlist(t0)), lambda xs: lambda x: xs + [x]),
            Primitive("cons", arrow(t0, tlist(t0), tlist(t0)), _cons),
            Primitive("insert", arrow(t0, tint, tlist(t0), tlist(t0)), _insert),
            Primitive("++", arrow(tlist(t0), tlist(t0), tlist(t0)), _concat),
            Primitive("splice", arrow(tlist(t0), tint, tlist(t0), tlist(t0)), _splice),
            Primitive("first", arrow(tlist(t0), t0), lambda xs: xs[0]),
            Primitive("second", arrow(tlist(t0), t0), lambda xs: xs[1]),
            Primitive("third", arrow(tlist(t0), t0), lambda xs: xs[2]),
            Primitive("last", arrow(tlist(t0), t0), lambda xs: xs[-1]),
            Primitive("index", arrow(tint, tlist(t0), t0), _index),
            Primitive("replaceEl", arrow(tint, t0, tlist(t0), tlist(t0)), _replaceEl),
            Primitive("swap", arrow(tint, tint, tlist(t0), tlist(t0)), _swap),
            Primitive("cut_idx", arrow(tint, tlist(t0), tlist(t0)), _cutIdx),
            Primitive("cut_val", arrow(t0, tlist(t0), tlist(t0)), _cutVal),
            Primitive("cut_vals", arrow(t0, tlist(t0), tlist(t0)), _cutVals),
            Primitive("drop", arrow(tint, tlist(t0), tlist(t0)), _drop),
            Primitive("droplast", arrow(tint, tlist(t0), tlist(t0)), _droplast),
            Primitive("cut_slice", arrow(tint, tint, tlist(t0), tlist(t0)), _cutSlice),
            Primitive("take", arrow(tint, tlist(t0), tlist(t0)), _take),
            Primitive("takelast", arrow(tint, tlist(t0), tlist(t0)), _takelast),
            Primitive("slice", arrow(tint, tint, tlist(t0), tlist(t0)), _slice),
            Primitive("fold", arrow(tlist(t0), t1, arrow(t0, t1, t1), t1), _fold),
            Primitive("foldi", arrow(tlist(t0), t1, arrow(tint, t0, t1, t1), t1), _foldi),
            Primitive("filter", arrow(arrow(t0, tbool), tlist(t0), tlist(t0)), _filter),
            Primitive("filteri", arrow(arrow(tint, t0, tbool), tlist(t0), tlist(t0)), _filteri),
            Primitive("count", arrow(arrow(t0, tbool), tlist(t0), tint), _count),
            Primitive("find", arrow(arrow(t0, tbool), tlist(t0), tlist(tint)), _findAll),
            Primitive("map", arrow(arrow(t0, t1), tlist(t0), tlist(t1)), _map),
            Primitive("mapi", arrow(arrow(tint, t0, t1), tlist(t0), tlist(t1)), _mapi),
            Primitive("group", arrow(arrow(t0, t1), tlist(t0), tlist(tlist(t0))), _group),
            Primitive("is_in", arrow(tlist(t0), t0, tbool), lambda l: lambda x: x in l),
            Primitive("length", arrow(tlist(t0), tint), len),
            Primitive("max", arrow(tlist(tint), tint), max),
            Primitive("min", arrow(tlist(tint), tint), min),
            Primitive("product", arrow(tlist(tint), tint), _product),
            Primitive("sum", arrow(tlist(tint), tint), sum),
            # Primitive("unique", arrow(tlist(t0), tlist(t0)), lambda l: list(OrderedDict.fromkeys(l))),
            Primitive("unique", arrow(tlist(t0), tlist(t0)), lambda l: list(set(l))),
            Primitive("sort", arrow(tlist(tint), tlist(tint)), sorted),
            Primitive("reverse", arrow(tlist(t0), tlist(t0)), _reverse),
            Primitive("flatten", arrow(tlist(tlist(t0)), tlist(t0)), _flatten),
            Primitive("zip", arrow(tlist(t0), tlist(t1), arrow(t0, t1, t2), tlist(t2)), _zip)
        ]


# list repeating x n times
def _repeat(x): return lambda n: [x for i in range(n)]

# list of numbers from i to j, inclusive, counting by n
def _rangeGeneral(i): 
    def g(j):
        if j < i:
            raise ValueError
        def f(n):
            if n > 100:
                raise ValueError
            return list(range(i,j+1,n))
        return f
    return g

# insert x at index i in xs
def _insert(x):
    def g(i):
        def f(xs):
            return xs[:i] + [x] + xs[i:]
        return f
    return g

# append x to xs
def _append(xs):
    def g(x):
        return xs + [x]
    return g

# concatenate xs and ys
def _concat(xs): return lambda ys: xs + ys

# insert ys into xs, beginning at index i
def _splice(ys):
    def g(i):
        def f(xs):
            return xs[:i] + ys + xs[i:]
        return f
    return g

# replace element at index i in xs with x
def _replaceEl(i):
    def g(x):
        def f(xs):
            return xs[:i] + [x] + xs[i+1:]
        return f
    return g

# swap elements at indices i and j in xs
def _swap(i):
    def g(j):
        def f(xs):
            xsCopy = xs[::]
            temp = xsCopy[i]
            xsCopy[i] = xsCopy[j]
            xsCopy[j] = temp
            return xsCopy
        return f
    return g

# remove element at index i from xs (returns cs if x not in xs)
def _cutIdx(i): 
    def g(xs):
        if i >= len(xs):
            return xs
        elif i == len(xs)-1:
            return xs[:i]
        else:
            return xs[:i] + xs[i+1:]
    return g

# remove first occurence of x from xs (returns xs if x not in xs)
def _cutVal(x):
    def g(xs):
        xsCopy = xs[::]
        if x not in xsCopy:
            return xsCopy
        else:
            xsCopy.remove(x)
        return xsCopy
    return g


# remove all occurences of x from xs (returns xs if x not in xs)
def _cutVals(x): return lambda xs: [el for el in xs if el != x]

# remove first n elements from xs
def _drop(n):
    def g(xs):
        if n > len(xs)-1:
            return []
        else:
            return xs[n:]
    return g

# remove last n elements from xs
def _droplast(n):
    def g(xs):
        if n > len(xs)-1:
            return []
        else:
            return xs[:len(xs)-n]
    return g

# remove elements at indices i to j, inclusive from xs
def _cutSlice(i):
    def g(j):
        def f(xs):
            if j < i:
                raise ValueError
            elif j > len(xs):
                return xs[:i]
            else:
                return xs[:i] + xs[j+1:]
        return f
    return g

# take first n elements from xs
def _take(n):
    def g(xs):
        if n > len(xs)-1:
            return xs
        else:
            return xs[:n]
    return g

# take last n elements from xs
def _takelast(n):
    def g(xs):
        if n > len(xs)-1:
            return xs
        else:
            return xs[len(xs)-n:]
    return g

# count number of elements of xs for which f(x) is true
def _count(f):
    def g(xs):
        numElements = 0
        for el in xs:
            if f(el):
                numElements += 1
        return numElements
    return g

# return indices of xs for which p is true
def _findAll(f):
    def g(xs):
        indices = []
        for i,el in enumerate(xs):
            if f(el):
                indices.append(i)
        return indices
    return g

# group elements, x, of xs based on the key (f x)
def _group(f):
    def g(xs):
        groups = {}
        for x in xs:
            key = f(x)
            groups[key] = groups.get(key, []) + [x]
        return list(groups.values())
    return g

# product of elements in xs
def _product(l): return reduce(lambda a, x: a * x, l, 1)


def McCarthyPrimitives():
    "These are < primitives provided by 1959 lisp as introduced by McCarthy"
    return [
        Primitive("empty", tlist(t0), []),
        Primitive("cons", arrow(t0, tlist(t0), tlist(t0)), _cons),
        Primitive("car", arrow(tlist(t0), t0), _car),
        Primitive("cdr", arrow(tlist(t0), tlist(t0)), _cdr),
        Primitive("empty?", arrow(tlist(t0), tbool), _isEmpty),
        #Primitive("unfold", arrow(t0, arrow(t0,t1), arrow(t0,t0), arrow(t0,tbool), tlist(t1)), _isEmpty),
        #Primitive("1+", arrow(tint,tint),None),
        # Primitive("range", arrow(tint, tlist(tint)), range),
        # Primitive("map", arrow(arrow(t0, t1), tlist(t0), tlist(t1)), _map),
        # Primitive("index", arrow(tint,tlist(t0),t0),None),
        # Primitive("length", arrow(tlist(t0),tint),None),
        primitiveRecursion1,
        #primitiveRecursion2,
        Primitive("gt?", arrow(tint, tint, tbool), _gt),
        Primitive("if", arrow(tbool, t0, t0, t0), _if),
        Primitive("eq?", arrow(tint, tint, tbool), _eq),
        Primitive("+", arrow(tint, tint, tint), _addition),
        Primitive("-", arrow(tint, tint, tint), _subtraction),
    ] + [Primitive(str(j), tint, j) for j in range(2)]

def getPrimitive(primitiveList, primitiveName):
    return [prim for prim in primitiveList if prim.name == primitiveName]

def test_josh_rich_primitives():

    assert _repeat(4)(6) == [4,4,4,4,4,4]

    assert _rangeGeneral(1)(10)(2) == [1,3,5,7,9]
    assert _rangeGeneral(3)(12)(3) == [3,6,9,12]

    assert _insert(5)(2)([1,2,3]) == [1,2,5,3]


    assert _append([1,2,3])(6) == [1,2,3,6]

    assert _concat([])([]) == []
    assert _concat([5,6])([1]) == [5,6,1]

    assert _splice([1,2])(1)([9,7,8]) == [9,1,2,7,8]

    assert _index(2)([5,6,7]) == 7

    assert _replaceEl(1)(3)([5,6,7]) == [5,3,7]

    assert _swap(1)(2)([5,6,7]) == [5,7,6]

    assert _cutIdx(2)([5,6,7]) == [5,6]

    assert _cutVal(2)([5,6,2,7,2]) == [5,6,7,2]
    assert _cutVal(3)([1,2]) == [1,2]

    assert _cutVals(2)([5,6,2,7,2]) == [5,6,7]
    assert _cutVals(3)([1,2]) == [1,2]

    assert _drop(3)([1,2,3,4]) == [4]
    assert _drop(2)([1,2]) == []
    assert _drop(2)([1]) == []

    assert _droplast(3)([1,2,3,4]) == [1]
    assert _droplast(2)([1,2]) == []
    assert _droplast(2)([1]) == []

    assert _take(3)([1,2,3,4]) == [1,2,3]
    assert _take(10)([1,2,3,4]) == [1,2,3,4]

    assert _takelast(3)([1,2,3,4]) == [2,3,4]
    assert _takelast(10)([1,2,3,4]) == [1,2,3,4]

    assert _cutSlice(1)(3)([5,6,7,8,9]) == [5,9]
    assert _cutSlice(2)(9)([1,2,3,4]) == [1,2]
    assert _cutSlice(0)(5)([1,2]) == []

    # add the elements of the odd indices
    assert _foldi([7,2,3,4])(0)(lambda idx: lambda x: lambda a: a + x if (idx % 2 == 1) else a == 6)

    # remove the even index elements from list
    assert _filteri(lambda idx: lambda x: idx%2 == 0)([7,3,5,10]) == [3,10]

    assert _count(lambda x: x % 2 == 0)([3,7,5,2,4]) == 2

    assert _findAll(lambda x: x % 2 == 0)([3,7,5,2,4]) == [3,4]

    groups = _group(lambda x: x % 3)([1,2,3,4,5,6,7,8])
    assert sorted(groups, key=lambda group: group[0]) == [[1,4,7], [2,5,8], [3,6]]

    assert _product([5,6,7]) == 210

if __name__ == "__main__":
    pass


    # bootstrapTarget()
    # g = Grammar.uniform(McCarthyPrimitives())
    # # with open("/home/ellisk/om/ec/experimentOutputs/list_aic=1.0_arity=3_ET=1800_expandFrontier=2.0_it=4_likelihoodModel=all-or-nothing_MF=5_baseline=False_pc=10.0_L=1.0_K=5_rec=False.pickle", "rb") as handle:
    # #     b = pickle.load(handle).grammars[-1]
    # # print b

    # p = Program.parse(
    #     "(lambda (lambda (lambda (if (empty? $0) empty (cons (+ (car $1) (car $0)) ($2 (cdr $1) (cdr $0)))))))")
    # t = arrow(tlist(tint), tlist(tint), tlist(tint))  # ,tlist(tbool))
    # print(g.logLikelihood(arrow(t, t), p))
    # assert False
    # print(b.logLikelihood(arrow(t, t), p))

    # # p = Program.parse("""(lambda (lambda
    # # (unfold 0
    # # (lambda (+ (index $0 $2) (index $0 $1)))
    # # (lambda (1+ $0))
    # # (lambda (eq? $0 (length $1))))))
    # # """)
    # p = Program.parse("""(lambda (lambda
    # (map (lambda (+ (index $0 $2) (index $0 $1))) (range (length $0))  )))""")
    # # .replace("unfold", "#(lambda (lambda (lambda (lambda (fix1 $0 (lambda (lambda (#(lambda (lambda (lambda (if $0 empty (cons $1 $2))))) ($1 ($3 $0)) ($4 $0) ($5 $0)))))))))").\
    # # replace("length", "#(lambda (fix1 $0 (lambda (lambda (if (empty? $0) 0 (+ ($1 (cdr $0)) 1))))))").\
    # # replace("forloop", "(#(lambda (lambda (lambda (lambda (fix1 $0 (lambda (lambda (#(lambda (lambda (lambda (if $0 empty (cons $1 $2))))) ($1 ($3 $0)) ($4 $0) ($5 $0))))))))) (lambda (#(eq? 0) $0)) $0 (lambda (#(lambda (- $0 1)) $0)))").\
    # # replace("inc","#(lambda (+ $0 1))").\
    # # replace("drop","#(lambda (lambda (fix2 $0 $1 (lambda (lambda (lambda (if
    # # (#(eq? 0) $1) $0 (cdr ($2 (- $1 1) $0)))))))))"))
    # print(p)
    # print(g.logLikelihood(t, p))
    # assert False

    # print("??")
    # p = Program.parse(
    #     "#(lambda (#(lambda (lambda (lambda (fix1 $0 (lambda (lambda (if (empty? $0) $3 ($4 (car $0) ($1 (cdr $0)))))))))) (lambda $1) 1))")
    # for j in range(10):
    #     l = list(range(j))
    #     print(l, p.evaluate([])(lambda x: x * 2)(l))
    #     print()
    # print()

    # print("multiply")
    # p = Program.parse(
    #     "(lambda (lambda (lambda (if (eq? $0 0) 0 (+ $1 ($2 $1 (- $0 1)))))))")
    # print(g.logLikelihood(arrow(arrow(tint, tint, tint), tint, tint, tint), p))
    # print()

    # print("take until 0")
    # p = Program.parse("(lambda (lambda (if (eq? $1 0) empty (cons $1 $0))))")
    # print(g.logLikelihood(arrow(tint, tlist(tint), tlist(tint)), p))
    # print()

    # print("countdown primitive")
    # p = Program.parse(
    #     "(lambda (lambda (if (eq? $0 0) empty (cons (+ $0 1) ($1 (- $0 1))))))")
    # print(
    #     g.logLikelihood(
    #         arrow(
    #             arrow(
    #                 tint, tlist(tint)), arrow(
    #                 tint, tlist(tint))), p))
    # print(_fix(9)(p.evaluate([])))
    # print("countdown w/ better primitives")
    # p = Program.parse(
    #     "(lambda (lambda (if (eq0 $0) empty (cons (+1 $0) ($1 (-1 $0))))))")
    # print(
    #     g.logLikelihood(
    #         arrow(
    #             arrow(
    #                 tint, tlist(tint)), arrow(
    #                 tint, tlist(tint))), p))

    # print()

    # print("prepend zeros")
    # p = Program.parse(
    #     "(lambda (lambda (lambda (if (eq? $1 0) $0 (cons 0 ($2 (- $1 1) $0))))))")
    # print(
    #     g.logLikelihood(
    #         arrow(
    #             arrow(
    #                 tint,
    #                 tlist(tint),
    #                 tlist(tint)),
    #             tint,
    #             tlist(tint),
    #             tlist(tint)),
    #         p))
    # print()
    # assert False

    # p = Program.parse(
    #     "(lambda (fix1 $0 (lambda (lambda (if (empty? $0) 0 (+ 1 ($1 (cdr $0))))))))")
    # print(p.evaluate([])(list(range(17))))
    # print(g.logLikelihood(arrow(tlist(tbool), tint), p))

    # p = Program.parse(
    #     "(lambda (lambda (if (empty? $0) 0 (+ 1 ($1 (cdr $0))))))")
    # print(
    #     g.logLikelihood(
    #         arrow(
    #             arrow(
    #                 tlist(tbool), tint), arrow(
    #                 tlist(tbool), tint)), p))

    # p = Program.parse(
    #     "(lambda (fix1 $0 (lambda (lambda (if (empty? $0) 0 (+ (car $0) ($1 (cdr $0))))))))")

    # print(p.evaluate([])(list(range(4))))
    # print(g.logLikelihood(arrow(tlist(tint), tint), p))

    # p = Program.parse(
    #     "(lambda (lambda (if (empty? $0) 0 (+ (car $0) ($1 (cdr $0))))))")
    # print(p)
    # print(
    #     g.logLikelihood(
    #         arrow(
    #             arrow(
    #                 tlist(tint),
    #                 tint),
    #             tlist(tint),
    #             tint),
    #         p))

    # print("take")
    # p = Program.parse(
    #     "(lambda (lambda (lambda (if (eq? $1 0) empty (cons (car $0) ($2 (- $1 1) (cdr $0)))))))")
    # print(p)
    # print(
    #     g.logLikelihood(
    #         arrow(
    #             arrow(
    #                 tint,
    #                 tlist(tint),
    #                 tlist(tint)),
    #             tint,
    #             tlist(tint),
    #             tlist(tint)),
    #         p))
    # assert False

    # print(p.evaluate([])(list(range(4))))
    # print(g.logLikelihood(arrow(tlist(tint), tlist(tint)), p))

    # p = Program.parse(
    #     """(lambda (fix (lambda (lambda (match $0 0 (lambda (lambda (+ $1 ($3 $0))))))) $0))""")
    # print(p.evaluate([])(list(range(4))))
    # print(g.logLikelihood(arrow(tlist(tint), tint), p))
