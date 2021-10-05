from dreamcoder.program import Program
from dreamcoder.type import Context
from dreamcoder.utilities import get_root_dir

import copy
import dill
import json
import subprocess

class PartialProgram:
    def __init__(self, 
        primitiveToIdx,
        requestReturnType,
        device,
        programTokenSeq=None, 
        programStringsSeq=None, 
        nextTokenTypeStack=None, 
        lambdaVarsTypeStack=None, 
        openParenthesisStack=None,
        parentTokenStack=None,
        totalScore=None,
        hidden=None):

        self.primitiveToIdx = primitiveToIdx
        self.requestReturnType = requestReturnType
        self.device = device

        if programTokenSeq is None:
            self.programTokenSeq = [torch.tensor([primitiveToIdx["START"]], device=device)]
        else:
            self.programTokenSeq = programTokenSeq

        self.programStringsSeq = ["(lambda"] if programStringsSeq is None else programStringsSeq
        
        self.totalScore = 0.0 if totalScore is None else totalScore

        self.nextTokenTypeStack = Stack([requestReturnType]) if nextTokenTypeStack is None else nextTokenTypeStack
        self.lambdaVarsTypeStack = Stack() if lambdaVarsTypeStack is None else lambdaVarsTypeStack
        self.openParenthesisStack = Stack() if openParenthesisStack is None else openParenthesisStack
        if parentTokenStack is None:
            self.parentTokenStack = Stack([torch.tensor([primitiveToIdx["START"]], device=device)])
        else:
            self.parentTokenStack = parentTokenStack

        self.hidden = hidden

    def __lt__(self, other):
        return (self.totalScore < other.totalScore)

    def processNextToken(self, nextToken, nextTokenType, score, lambdaVars, primitiveToIdx, hidden, device):

        if nextToken == "START":
            assert Exception("Should never sample START")
        elif nextToken == "INPUT":
            self.programStringsSeq.append("${}".format(len(self.lambdaVarsTypeStack)))
            pass
        elif nextToken == "LAMBDA_INPUT":
            # TODO: fix so that we don't deterministically choose the first one
            self.programStringsSeq.append("${}".format(lambdaVars[0]))
            # lambdaVarsTypeStack.pop()
        elif nextToken == "LAMBDA":
            # assume lambdas are used only with one argument
            assert len(nextTokenType.functionArguments()) == 1
            for arg in nextTokenType.functionArguments():
                self.lambdaVarsTypeStack.push(arg)
            self.nextTokenTypeStack.push(nextTokenType.returns())
            # boolean stores whether this parenthesis corresponds to a lambda which is needed to manage LAMBDA_INPUT score
            self.openParenthesisStack.push((len(self.parentTokenStack), True))
            # technically the parent token is LAMBDA but that carries no information so we use the grandparent
            self.parentTokenStack.push(torch.tensor([primitiveToIdx["LAMBDA"]], device=device))
            self.programStringsSeq.append("(lambda")
        elif nextToken == "PAD":
            raise Exception("Should never get here since we are tracking program types PAD should always be masked")     
        else:
            sampledToken = Program.parse(nextToken)
            if sampledToken.tp.isArrow() and not(nextTokenType.isArrow()):
                # print("sampled function when desired type is constant -> add function arguments to type stack")
                self.programStringsSeq.append("(")
                # keep track of how many open left parenthesis they are by storing length of the parent token stack when its created
                # boolean stores whether this parenthesis corresponds to a lambda which is needed to manage LAMBDA_INPUT score
                self.openParenthesisStack.push((len(self.parentTokenStack), False))
                for arg in sampledToken.tp.functionArguments()[::-1]:
                    self.nextTokenTypeStack.push(arg)
                    # keep track of parent function for which existing one is an argument
                    self.parentTokenStack.push(primitiveToIdx[nextToken])
            self.programStringsSeq.append(str(sampledToken))

        self.programTokenSeq.append(primitiveToIdx[nextToken])

        # the openParenthesisStack will always contain numbers in ascending order
        while len(self.openParenthesisStack) > 0 and len(self.parentTokenStack) <= self.openParenthesisStack.toPop()[0]:
            self.programStringsSeq.append(")")
            _, isLambda = self.openParenthesisStack.pop()
            # if we are closing the scope of a lambda, we want to remove the LAMBDA_INPUT from the scope
            # poppedLambda ensures we only do this once
            if isLambda:
                self.lambdaVarsTypeStack.pop()

        self.hidden = hidden
        self.totalScore += score.item()
        # print("{}: {}".format(" ".join(self.programStringsSeq), self.totalScore))

        return self

    def detach_tensors(self):
            self.programTokenSeq = self.programTokenSeq.detach()

    def copy(self):
        return PartialProgram( 
            primitiveToIdx = self.primitiveToIdx,
            requestReturnType = self.requestReturnType,
            device = self.device,
            programTokenSeq = self.programTokenSeq.copy(), 
            programStringsSeq = self.programStringsSeq.copy(), 
            nextTokenTypeStack = self.nextTokenTypeStack.copy(), 
            lambdaVarsTypeStack = self.lambdaVarsTypeStack.copy(), 
            openParenthesisStack = self.openParenthesisStack.copy(),
            parentTokenStack = self.parentTokenStack.copy(),
            totalScore = self.totalScore,
            hidden = self.hidden)

class Stack:
    """
    Stack data structure to enforce type constraints when decoding a program by sequentially sampling
    its primitives
    """

    def __init__(self, stack=None):
        self.stack = [] if stack is None else stack

    def pop(self):
        return self.stack.pop(-1)

    def push(self, tp):
        self.stack.append(tp)
        return

    def toPop(self):
        if len(self.stack) > 0:
            return self.stack[-1]
        return None

    def copy(self):
        copiedList = self.stack.copy()
        return Stack(stack=copiedList)

    def __contains__(self, x):
        return x in self.stack

    def __len__(self):
        return len(self.stack)

    def __repr__(self):
        return self.stack.__str__()

    def __iter__(self):
        for x in self.stack[::-1]:
            yield x

def taskMessage(t, task_to_programs):
    m = {
        "examples": [{"inputs": [xs[0].toJson()], "output": y.toJson()} for xs, y in t.examples],
        "name": t.name,
        "request": t.request.json(),
        "programs": [el[0] for el in task_to_programs[t.name]]
    }
    return m

def execute_programs(tasks, grammar, task_to_programs):

    message = {
        "tasks": [taskMessage(t, task_to_programs) for t in tasks],
        "programTimeout": 0.1,
    }
    dumped_message = json.dumps(message)
    with open('message', 'w') as outfile:
        json.dump(message, outfile) 

    try:
        solver_file = os.path.join(get_root_dir(), "solvers/exec_arc_p")
        process = subprocess.Popen(
            solver_file, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        response, error = process.communicate(bytes(dumped_message, encoding="utf-8"))
        response = json.loads(response.decode("utf-8"))
        return response
        
    except OSError as exc:
        raise exc

def get_primitives_of_type(request, grammar):
	primitives = []
	for p in grammar.primitives:
		if p.tp.returns() == request:
			primitives.append(str(p))
	return primitives

def program_to_token_sequence(program, grammar):
	program_token_sequence = [token for token in program.left_order_tokens_alt()]
	return ["START"] + program_token_sequence


# Uses `parameters` to construct the checkpoint path
def checkpointPath(iteration, extra=""):
    parameters["iterations"] = iteration
    kvs = [
        "{}={}".format(
            ECResult.abbreviate(k),
            parameters[k]) for k in sorted(
            parameters.keys())]
    return "{}_{}{}.pickle".format(outputPrefix, "_".join(kvs), extra)

def resume_from_path(resume):
    try:
        resume = int(resume)
        path = checkpointPath(resume)
    except ValueError:
        path = resume
    with open(path, "rb") as handle:
        result = dill.load(handle)
    resume = len(result.grammars) - 1
    print("Loaded checkpoint from", path)
    grammar = result.grammars[-1] if result.grammars else grammar
    return result, grammar, result.grammars[0]

def main():

	path = "experimentOutputs/arc/2021-04-21T20:11:03.040547/"
	pickleFilename = "arc_arity=0_BO=False_CO=True_ES=1_ET=600_t_zero=3600_HR=0_it=2_MF=10_noConsolidation=True_RS=10000_RT=3600_RR=False_RW=False_solver=ocaml_STM=True_TRR=default_K=2_topkNotMAP=False_DSL=False.pickle"
	picklePath1 = path + pickleFilename
	result, _, _ = resume_from_path(picklePath1)

	for t,f in result.allFrontiers.items():
		if len(f.entries) > 0:
			print(f.topK(1).entries[0].program.size())

		for i,o in t.examples:
			print(i[0].toJson())
			print(o.toJson())
