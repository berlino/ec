import torch

from larc.decoderUtils import *
import heapq

MAX_PROGRAM_LENGTH = 20

def randomized_beam_search_decode(decoder, encoderOutput, beam_width, epsilon, device):

    num_end_nodes = beam_width

    # list of final programs
    endNodes = []

    # starting node
    nodes = [PartialProgram(decoder.primitiveToIdx, decoder.request.returns(), device)]

    while True:
        newNodes = []
        # print("\nExpanding Beam")
        for k in range(beam_width):
            # true with probability epsilon and false with probability (1-epsilon)
            if len(nodes) == 0:
                 if len(newNodes) == 0:
                     return endNodes
                 continue
            if random.random() < epsilon:
                 i = random.randint(0,len(nodes)-1)
                 node = nodes[i]
                 nodes[i] = nodes[0]
                 heapq.heappop(nodes)
            else:
                 node = heapq.heappop(nodes)
                 # print("Selected node has score: {}".format(node.totalScore))
        
            
            # print("{} total nodes, Selected node has {} tokens, {} score, {} endNodes found".format(len(newNodes), len(node.programTokenSeq), node.totalScore, len(endNodes)))
            # print("Selected node: {}".format(node.programStringsSeq))
            attnOutputWeights, nextTokenType, lambdaVars, node = decoder.forward(encoderOutput, node, node.parentTokenStack.pop(),
device=device, restrictTypes=True)
            # print('pp', node.programStringsSeq)
            # print("nextTokenType", nextTokenType)
            # print(attnOutputWeights)
            attnOutputWeights = attnOutputWeights[0, 0, :]

            weights, indices = attnOutputWeights[attnOutputWeights > 0], attnOutputWeights.nonzero()
            # print(indices)
            # print([decoder.idxToPrimitive[idx.item()] for idx in indices])

            # add to queue
            for idx in indices:
                nextToken = decoder.idxToPrimitive[idx.item()]
                newNode = node.copy()
                nllScore = -torch.log(attnOutputWeights[idx])
                newNode.processNextToken(nextToken, nextTokenType, nllScore, lambdaVars, decoder.primitiveToIdx, device)

                if len(newNode.nextTokenTypeStack) == 0:
                    endNodes.append((newNode.totalScore, newNode))
                    # print("{} total nodes, Selected node has {} tokens, {} endNodes found".format(len(nodes), len(newNode.programTokenSeq), len(endNodes)))
                    # print("Selected node: {}".format(newNode.programStringsSeq))
                    if len(endNodes) >= num_end_nodes:
                        return endNodes
                
                # using the type system we can be intelligent about ruling out programs that don't satisfy
                elif len(newNode.programTokenSeq) + len(newNode.nextTokenTypeStack) > MAX_PROGRAM_LENGTH:
                    continue
                else:
                    heapq.heappush(newNodes, newNode)
        nodes = newNodes 
    return endNodes


